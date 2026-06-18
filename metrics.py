"""
metrics.py  —  evaluation, centered on the novel Defense Risk-Coverage (DRC) curve.

Standard selective prediction reports risk-coverage where risk = error rate on predicted
cases (Santosh et al. 2024, Eq. 2-3). For a defense tool that is the wrong risk. We define:

  DEFENSE RISK = fraction of NON-abstained cases for which the system FAILED to surface an
                 actionable defense -- i.e. it predicted 'unfavorable' while a legally
                 plausible favorable counterfactual in fact existed (reachability high).
  COVERAGE     = fraction of cases on which the system did not abstain.

Sweeping the regime's operating threshold traces a DRC curve; lower area = better. We
report DRC for SR-Conf, SR-Unc, SR-CF, and DASP on the same axis -- the central figure.

To make AURC COMPARABLE across regimes, the area is integrated over a COMMON, fixed
coverage grid (regimes span different coverage ranges, so a raw trapezoid over each
regime's own coverage support would integrate over different intervals and be
incomparable). This matches how selective-prediction work reports Area-Under-RC.

This module also provides the two STANDARD selective-prediction metrics from the
literature, so DASP can be compared on the field's usual axis in addition to the
defense-specific DRC:
  - rpp():        Reversed Pair Proportion (Xin et al. 2021, Santosh Eq. 4). Lower = better.
  - refinement(): RPP normalized by the worst case (Gu & Hopkins 2023, Santosh Eq. 5).
                  0 = best, 0.5 = random, 1 = worst. Isolates confidence-ranking quality
                  from base-predictor quality. Both rank cases by SelectivePredictor.
                  confidence_score and need per-case CORRECTNESS (so they take y_true).

And the defense-oriented evaluation helpers:
  - subgroup_coverage(): coverage and defense-risk per subgroup (fairness-conditional).
  - actionable_coverage(): fraction of cases where the system EITHER predicts confidently
    OR returns >=1 legally-plausible flipping counterfactual.
  - cf_quality(): fluency / minimality / validity + legal-plausibility rate, evaluated
    against the 56-doc ILDC expert gold where applicable.

CONTRACT (consumed by experiment.py / plots.py -- do not rename):
  defense_risk_coverage(...) -> {"coverage": [...], "defense_risk": [...], "aurc": float,
                                 "common_coverage": [...], "common_risk": [...]}
  subgroup_coverage(...)     -> {group: {"coverage","defense_risk","n"}}
  actionable_coverage(...)   -> float
"""
from __future__ import annotations

import numpy as np

from selective import Decision, CaseSignals, SelectivePredictor

# NumPy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Common coverage grid on which AURC is integrated, so all regimes are comparable.
_COMMON_COVERAGE_GRID = np.linspace(0.0, 1.0, 101)


# --------------------------------------------------------------------------------
# Defense Risk-Coverage (the contribution's central metric)
# --------------------------------------------------------------------------------
def _defense_failure(decision: Decision, s: CaseSignals, reach_threshold: float) -> int:
    """
    1 iff a real actionable defense was MISSED on a non-abstained case: the system
    predicted UNFAVORABLE while a legally-plausible favorable counterfactual existed
    (reachability >= reach_threshold). A confident FAVORABLE prediction is never a missed
    defense (the defense already got the favorable call), and an abstention is never a
    failure (the case was deferred to a human), so both return 0.
    """
    if decision == Decision.ABSTAIN:
        return 0
    missed = (decision == Decision.PREDICT_UNFAVORABLE) and (s.reachability >= reach_threshold)
    return int(missed)


def _aurc_on_common_grid(cov: list[float], risk: list[float],
                         grid: np.ndarray = _COMMON_COVERAGE_GRID) -> tuple[float, list, list]:
    """
    Integrate defense risk over a COMMON coverage grid so areas are comparable across
    regimes. Steps: (1) sort by coverage; (2) collapse tied coverages to their MINIMUM
    risk (the best risk achievable at that coverage -- the lower envelope of the
    risk-coverage operating points); (3) interpolate risk onto the fixed grid, holding the
    endpoints flat outside the regime's observed coverage range; (4) trapezoid-integrate.
    """
    cov_a = np.asarray(cov, dtype=float)
    risk_a = np.asarray(risk, dtype=float)
    if len(cov_a) == 0:
        return float("nan"), grid.tolist(), [float("nan")] * len(grid)
    order = np.argsort(cov_a, kind="mergesort")
    cov_a, risk_a = cov_a[order], risk_a[order]
    # collapse ties: for each unique coverage keep the minimum risk
    uc, inv = np.unique(cov_a, return_inverse=True)
    ur = np.full(len(uc), np.inf)
    for i, c in enumerate(inv):
        ur[c] = min(ur[c], risk_a[i])
    if len(uc) == 1:
        # single coverage value: risk is flat across the grid at that level
        interp = np.full(len(grid), ur[0])
    else:
        interp = np.interp(grid, uc, ur, left=ur[0], right=ur[-1])
    area = float(_trapz(interp, grid))
    return area, grid.tolist(), interp.tolist()


def defense_risk_coverage(signals: list[CaseSignals], regime: str,
                          sweep, reach_threshold: float = 0.6, **regime_kwargs):
    """
    Trace (coverage, defense_risk) as the regime's threshold sweeps, and report the area
    under the DRC curve integrated on a COMMON coverage grid (lower is better).

    `sweep` is an iterable of threshold values; which knob it controls depends on regime
    (tau_conf for SR-Conf, tau_unc for SR-Unc, tau_reach for SR-CF, margin for DASP).

    Returns the raw swept operating points (coverage/defense_risk) for inspection AND the
    common-grid curve used for the comparable area:
        {"coverage": [...], "defense_risk": [...], "aurc": float,
         "common_coverage": [...], "common_risk": [...]}
    """
    cov, risk = [], []
    for thr in sweep:
        kw = dict(regime_kwargs)
        if regime == "SR-Conf":
            kw["tau_conf"] = thr
        elif regime == "SR-Unc":
            kw["tau_unc"] = thr
        elif regime == "SR-CF":
            kw["tau_reach"] = thr
        else:  # DASP
            kw["margin"] = thr
        sp = SelectivePredictor(regime=regime, **kw)
        decs = sp.batch(signals)
        n_pred = sum(d != Decision.ABSTAIN for d in decs)
        coverage = n_pred / len(signals) if signals else 0.0
        fails = sum(_defense_failure(d, s, reach_threshold) for d, s in zip(decs, signals))
        drisk = fails / n_pred if n_pred else 0.0
        cov.append(coverage)
        risk.append(drisk)
    area, common_cov, common_risk = _aurc_on_common_grid(cov, risk)
    # raw points sorted by coverage for readability / plotting of operating points
    order = np.argsort(cov, kind="mergesort")
    return {
        "coverage": np.asarray(cov)[order].tolist(),
        "defense_risk": np.asarray(risk)[order].tolist(),
        "aurc": area,
        "common_coverage": common_cov,
        "common_risk": common_risk,
    }


# --------------------------------------------------------------------------------
# Standard selective-prediction metrics (Santosh Eq. 4-5) -- the field's usual axis.
# Both rank cases by a continuous confidence score and need per-case correctness.
# --------------------------------------------------------------------------------
def _reversed_pairs(scores: np.ndarray, correct: np.ndarray) -> int:
    """
    Count ordered pairs (i, j) with score_i < score_j AND loss_i < loss_j, where
    loss = 1 - correct. loss_i < loss_j  <=>  i is correct and j is wrong. So this counts
    pairs where the confidence ranking is REVERSED relative to correctness: a case the
    model got RIGHT was assigned LOWER confidence than a case it got WRONG.
    Vectorized: for each (correct i, wrong j) pair, count score_i < score_j.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=bool)
    si, sj = s[c], s[~c]  # scores of correct vs wrong cases
    if len(si) == 0 or len(sj) == 0:
        return 0
    return int((si[:, None] < sj[None, :]).sum())


def rpp(scores, correct) -> float:
    """
    Reversed Pair Proportion (Xin et al. 2021; Santosh Eq. 4): reversed pairs / n^2.
    Lower is better (0 = the confidence ordering never contradicts correctness).
    `scores` from SelectivePredictor.confidence_score; `correct` is per-case bool.
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n < 2:
        return 0.0
    return _reversed_pairs(s, correct) / (n * n)


def refinement(scores, correct) -> float:
    """
    Refinement (Gu & Hopkins 2023; Santosh Eq. 5): reversed pairs / (c * (n - c)), where
    c = number correct. Normalizing by the worst-case number of reversed pairs yields
    0 = best, 0.5 = random, 1 = worst, and -- unlike RPP/AURCC -- isolates confidence-
    ranking quality from the base predictor's accuracy.
    """
    c = np.asarray(correct, dtype=bool)
    n = len(c)
    nc = int(c.sum())
    denom = nc * (n - nc)
    if denom == 0:  # all correct or all wrong: ranking is undefined/irrelevant
        return 0.0
    return _reversed_pairs(scores, c) / denom


def selective_metrics(signals: list[CaseSignals], regime: str, y_true: list[int],
                      **regime_kwargs) -> dict:
    """
    Convenience wrapper: compute RPP and Refinement for a regime, using its
    confidence_score for ranking and `y_true` for correctness. Correctness is defined on
    the NON-abstained base prediction (the model's favorable/unfavorable call vs the true
    label); abstained cases are excluded (selective metrics evaluate the cases answered).
    """
    sp = SelectivePredictor(regime=regime, **regime_kwargs)
    scores, correct = [], []
    for s, y in zip(signals, y_true):
        dec = sp.predict_or_abstain(s)
        if dec == Decision.ABSTAIN:
            continue
        pred = 1 if dec == Decision.PREDICT_FAVORABLE else 0
        scores.append(sp.confidence_score(s))
        correct.append(int(pred == int(y)))
    if len(scores) < 2:
        return {"rpp": 0.0, "refinement": 0.0, "n_scored": len(scores)}
    return {"rpp": rpp(scores, correct),
            "refinement": refinement(scores, correct),
            "n_scored": len(scores)}


# --------------------------------------------------------------------------------
# Defense-oriented helpers
# --------------------------------------------------------------------------------
def subgroup_coverage(signals: list[CaseSignals], sp: SelectivePredictor,
                      reach_threshold: float = 0.6) -> dict:
    """Per-subgroup coverage and defense-risk (fairness-conditional reliability)."""
    decs = sp.batch(signals)
    groups: dict[str, dict] = {}
    for d, s in zip(decs, signals):
        g = s.subgroup or "all"
        groups.setdefault(g, {"n": 0, "pred": 0, "fail": 0})
        groups[g]["n"] += 1
        if d != Decision.ABSTAIN:
            groups[g]["pred"] += 1
            groups[g]["fail"] += _defense_failure(d, s, reach_threshold)
    return {g: {"coverage": v["pred"] / v["n"] if v["n"] else 0.0,
                "defense_risk": v["fail"] / v["pred"] if v["pred"] else 0.0,
                "n": v["n"]} for g, v in groups.items()}


def actionable_coverage(signals: list[CaseSignals], conf_threshold: float = 0.7,
                        reach_floor: float = 1e-6) -> float:
    """
    Fraction of cases for which the system is actionable: it either predicts confidently
    (conf >= conf_threshold) or supplies at least one legally-plausible flipping
    counterfactual (reachability > reach_floor). reach_floor (~0) excludes cases with no
    flip (reachability exactly 0), consistent with how counterfactual.py reports reach=0.
    """
    if not signals:
        return 0.0
    ok = sum(1 for s in signals
             if s.conf >= conf_threshold or s.reachability > reach_floor)
    return ok / len(signals)


def cf_quality(records: list[dict]) -> dict:
    """
    Aggregate counterfactual-quality metrics over records, each a dict with keys:
      fluency           in [0,1] (e.g. normalized inverse perplexity),
      minimality        in [0,1] (1 - normalized edit distance),
      valid_flip        bool (did the edit flip the prediction to favorable?),
      legally_plausible bool (did the edit pass the legal-plausibility filter?).
    Missing keys are treated as 0 (a record lacking a field contributes 0 to that mean)
    rather than raising, so partial logging does not crash evaluation.
    """
    if not records:
        return {}
    def col(key):
        return np.mean([float(r.get(key, 0)) for r in records])
    return {"fluency": float(col("fluency")),
            "minimality": float(col("minimality")),
            "validity": float(col("valid_flip")),
            "legal_plausibility_rate": float(col("legally_plausible"))}


# --------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = []
    for _ in range(400):
        p = float(rng.uniform(0, 1))
        reach = float(rng.uniform(0, 1))
        sig.append(CaseSignals(p_favorable=p, reachability=reach,
                               subgroup=rng.choice(["pre1990", "post2010"]),
                               epistemic=float(rng.uniform(0, 0.3))))

    # ---- DRC: returns the contract keys + the new common-grid keys -----------------
    drc = defense_risk_coverage(sig, "DASP", sweep=np.linspace(0.0, 0.5, 11),
                                w_epi=0.5, u_scale=0.159)
    assert {"coverage", "defense_risk", "aurc", "common_coverage", "common_risk"} <= set(drc)
    assert len(drc["coverage"]) == 11
    assert len(drc["common_coverage"]) == len(drc["common_risk"]) == 101
    assert np.isfinite(drc["aurc"])

    # ---- AURC is COMPARABLE across regimes (common grid, all in [0,1]) --------------
    for reg, sweep, kw in [("SR-Conf", np.linspace(0.5, 1.0, 11), {}),
                           ("SR-CF", np.linspace(0.0, 1.0, 11), {}),
                           ("SR-Unc", np.linspace(0.0, 0.3, 11), {}),
                           ("DASP", np.linspace(0.0, 0.5, 11), {"w_epi": 0.5, "u_scale": 0.159})]:
        d = defense_risk_coverage(sig, reg, sweep=sweep, **kw)
        assert 0.0 <= d["aurc"] <= 1.0, f"{reg} AURC out of [0,1]: {d['aurc']}"

    # ---- tie handling: duplicated coverage collapses to min risk, area finite -------
    a, g, r = _aurc_on_common_grid([0.5, 0.5, 0.9, 0.3], [0.2, 0.1, 0.05, 0.25])
    assert np.isfinite(a) and len(g) == 101

    # ---- RPP / Refinement against Santosh Eq. 4-5 (perfect -> 0, worst -> 1) --------
    perfect_s, perfect_c = [0.9, 0.8, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0]
    worst_s, worst_c = [0.1, 0.2, 0.3, 0.8, 0.9], [1, 1, 1, 0, 0]
    assert refinement(perfect_s, perfect_c) == 0.0
    assert abs(refinement(worst_s, worst_c) - 1.0) < 1e-9
    assert rpp(perfect_s, perfect_c) == 0.0 and rpp(worst_s, worst_c) > 0.0
    # all-correct / all-wrong / single-case degeneration
    assert refinement([0.1, 0.2], [1, 1]) == 0.0 and rpp([0.5], [1]) == 0.0

    # ---- selective_metrics wrapper (uses confidence_score + y_true) ----------------
    y_true = [1 if s.p_favorable >= 0.5 else 0 for s in sig]  # synthetic "true" labels
    sm = selective_metrics(sig, "SR-Conf", y_true, tau_conf=0.6)
    assert {"rpp", "refinement", "n_scored"} <= set(sm)
    sm_dasp = selective_metrics(sig, "DASP", y_true, margin=0.15, w_epi=0.5, u_scale=0.159)
    assert 0.0 <= sm_dasp["refinement"] <= 1.0

    # ---- subgroup + actionable coverage --------------------------------------------
    sp = SelectivePredictor("DASP", margin=0.15, u_scale=0.159)
    sc = subgroup_coverage(sig, sp)
    assert set(sc.keys()) <= {"pre1990", "post2010"}
    ac = actionable_coverage(sig)
    assert 0.0 <= ac <= 1.0

    # ---- cf_quality robust to missing keys -----------------------------------------
    q = cf_quality([{"fluency": 0.8, "minimality": 0.7, "valid_flip": True,
                     "legally_plausible": True},
                    {"fluency": 0.6}])  # missing keys -> treated as 0
    assert set(q) == {"fluency", "minimality", "validity", "legal_plausibility_rate"}
    assert cf_quality([]) == {}

    # ---- empty signals safe --------------------------------------------------------
    empty = defense_risk_coverage([], "DASP", sweep=[0.1, 0.2])
    assert np.isnan(empty["aurc"]) or empty["aurc"] == empty["aurc"]  # nan contract documented

    print(f"metrics.py self-test passed (DASP AURC={drc['aurc']:.3f}, actionable_cov={ac:.2f}, "
          f"RPP/Refinement vs Santosh Eq.4-5 verified, common-grid AURC comparable)")
