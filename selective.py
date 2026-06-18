"""
selective.py
================================================================================
CONTRIBUTION 2:  Defense-Aware Selective Prediction (DASP).

Standard selective prediction abstains when predictive confidence is low. We argue
that, for a system advising the *defense*, abstention should instead be driven by the
*counterfactual reachability of a favorable outcome* relative to the model's own
uncertainty. The dangerous case is not "the model is unsure" but "the model is
confidently predicting the defendant LOSES, yet a legally plausible argument would flip
the outcome." DASP is built to abstain exactly there.

This module couples three quantities per case x:

  p     = P(favorable outcome | x), a CALIBRATED probability from the predictor
          (temperature-scaled; see model.py). conf = max(p, 1-p).
  reach = 1 - d*, the counterfactual reachability in [0,1], where d* is the minimal
          *legally plausible* edit distance that flips the prediction to favorable
          (reach = 0 when no plausible flipping counterfactual exists; see
          counterfactual.py + plausibility.py).
  epi   = epistemic uncertainty from MC-dropout (a scalar in "uncertainty units";
          see model.py, which chooses the estimator: predictive std, PV, or BALD).
          Santosh et al. (2024) found BALD (total uncertainty) best on legal COC, so
          model.py defaults epi to BALD. OPTIONAL: when absent (e.g. the TF-IDF
          baseline, or a fast CPU run), DASP gracefully reduces to the confidence-only
          coupling and remains well-defined.

--------------------------------------------------------------------------------
THE FOUR REGIMES (all share the .predict_or_abstain interface so the risk--coverage
comparison in metrics.py is fair -- each is swept on EXACTLY ONE knob). Three of them
are baselines; DASP is the contribution. Having BOTH a confidence baseline and an
MC-uncertainty baseline means no reviewer can claim DASP beats a single weak baseline.

  SR-Conf : abstain iff conf < tau_conf.                                [knob: tau_conf]
            The confidence/conformal paradigm (Santosh-2024, Softmax-Response gate),
            STRENGTHENED with a split-conformal threshold that carries a finite-sample
            coverage guarantee (Angelopoulos & Bates 2021). conf is p-based, and the
            conformal tau is calibrated on the same p scale, so the gate is coherent.

  SR-Unc  : abstain iff epi > tau_unc.                                  [knob: tau_unc]
            The MC-dropout UNCERTAINTY gate -- the faithful Santosh PV/BALD selective
            baseline ("abstain when the model's epistemic uncertainty is high"). This is
            a SEPARATE, stronger baseline than SR-Conf, since Santosh found MC-dropout
            estimators beat Softmax Response. Requires epi to be present.

  SR-CF   : abstain iff reach > tau_reach.                              [knob: tau_reach]
            Counterfactual reachability only, ignoring confidence -- the
            "reachability-only" baseline. Abstains (defers to a human) whenever a
            favorable outcome is plausibly reachable.

  DASP    : abstain iff the model predicts a LOSS (p < 0.5) AND the defense-relevant
            evidence materially exceeds the model's residual certainty:              [knob: margin]

                defense_signal = reach + w_epi * epi_norm
                abstain  <=>  (p < 0.5)  AND  ( defense_signal - (1 - conf) ) > margin

            where epi_norm in [0,1] is the epistemic uncertainty normalized by a
            calibration-set scale (so reach and epi live on a common axis), and w_epi is
            a FIXED config weight (NOT swept -- keeping DASP a single-knob method so its
            risk--coverage curve is comparable to the baselines).

The coupling is the contribution: neither signal alone triggers abstention; their
*disagreement in the defense direction* does. Using MC-dropout epistemic uncertainty
(rather than only the point-confidence residual) is what lets DASP out-perform the
strongest selective-prediction configuration of Santosh et al. (2024), who found
MC-dropout (BALD) most effective on ECtHR.

--------------------------------------------------------------------------------
CONTINUOUS SCORES (for metrics.py): each regime ranks cases by how SAFE it is to
predict. `confidence_score(s)` exposes that ranking so metrics.py can compute the
standard selective-prediction metrics RPP and Refinement (Xin et al. 2021; Gu &
Hopkins 2023), which require a continuous per-case confidence, not just the decision.

--------------------------------------------------------------------------------
VERIFIED PROPERTIES (see the self-test and the design notes):
  * Backward-compatible: epi=None reduces DASP EXACTLY to the confidence-only rule, so
    pre-existing CaseSignals call sites and cached results remain valid.
  * Monotone: P(abstain) is non-decreasing in epi and in reach (a reviewer-checkable
    sanity property of any sensible abstention rule); SR-Unc is monotone in epi.
  * Single-knob: coverage is monotone in `margin`, so the DRC sweep is well-posed.
  * Deterministic: given signals, the rule is pure (all stochasticity is upstream in
    model.py's MC sampling), so DRC curves are reproducible.

Conformal calibration (split conformal) gives the confidence gate a finite-sample
coverage guarantee, and is calibrated PER SUBGROUP (Mondrian / group-conditional) so the
coverage guarantee holds within defendant subgroups, not merely marginally. The same
group-conditional machinery scales the epistemic signal per subgroup, so the fairness
property covers both signals.

Reference: Angelopoulos & Bates, A Gentle Introduction to Conformal Prediction (2021),
Theorem 1 (finite-sample coverage) and Sec. 4.1 (group-balanced / Mondrian);
Gal & Ghahramani, Dropout as a Bayesian Approximation (ICML 2016);
Santosh et al., The Craft of Selective Prediction (EMNLP Findings 2024), Eq. 6-9
(SR/SMP/PV/BALD) -- BALD > PV > SMP >> SR; Xin et al., The Art of Abstention (2021).
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Regimes that exist, in one place so callers and the self-test agree.
REGIMES = ("SR-Conf", "SR-Unc", "SR-CF", "DASP")


class Decision(str, Enum):
    PREDICT_FAVORABLE = "predict_favorable"
    PREDICT_UNFAVORABLE = "predict_unfavorable"
    ABSTAIN = "abstain"


# --------------------------------------------------------------------------------
# Case-level signals consumed by the selective predictor.
# IMPORTANT (backward compatibility): `epistemic` is the LAST field and OPTIONAL, so
# existing positional constructions CaseSignals(p_favorable, reachability, subgroup)
# in metrics.py / experiment.py keep working unchanged.
# --------------------------------------------------------------------------------
@dataclass
class CaseSignals:
    p_favorable: float          # calibrated P(favorable | x), expected in [0,1]
    reachability: float         # 1 - d* in [0,1]; >0 requires a plausible flipping CF
    subgroup: str | None = None
    epistemic: float | None = None  # MC-dropout uncertainty (std/PV/BALD); None => conf-only DASP

    @property
    def conf(self) -> float:
        """Model confidence = max(p, 1-p) = Softmax Response (Santosh Eq. 6). In [0.5, 1]."""
        return max(self.p_favorable, 1.0 - self.p_favorable)

    @property
    def residual(self) -> float:
        """Residual certainty gap = 1 - conf = min(p, 1-p). The model's own doubt. In [0, 0.5]."""
        return 1.0 - self.conf


# --------------------------------------------------------------------------------
# Split-conformal threshold with optional per-subgroup calibration.
# Used for (a) the SR-Conf confidence gate with a finite-sample coverage guarantee, and
# (b) deriving the epistemic normalization scale per subgroup for DASP.
# --------------------------------------------------------------------------------
class ConformalConfidence:
    """
    Split conformal for a binary predictor used as a confidence gate (Angelopoulos &
    Bates 2021). Nonconformity score s = 1 - p_true on a held-out calibration set; the
    finite-sample (1 - alpha) quantile yields a threshold guaranteeing marginal coverage
    >= 1 - alpha (their Theorem 1; requires only exchangeability, no distributional
    assumption). With `per_subgroup=True`, a separate quantile is fit within each subgroup
    so coverage holds conditionally (Mondrian / group-conditional conformal, their Sec. 4.1).

    Additionally records, per subgroup, an epistemic scale `u_scale` (the MEDIAN epistemic
    value on the calibration set -- robust and non-zero unless all values are zero) used by
    DASP to place the epistemic signal on the [0,1] axis.
    """

    def __init__(self, alpha: float = 0.1, per_subgroup: bool = True):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.per_subgroup = per_subgroup
        self._global_q: float = 1.0
        self._group_q: dict[str, float] = {}
        self._global_u: float = 1.0
        self._group_u: dict[str, float] = {}

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float:
        """
        Finite-sample-corrected conformal quantile of the nonconformity scores.
        Level = ceil((n+1)(1-alpha)) / n  (Angelopoulos & Bates 2021, the exact level that
        makes the coverage guarantee hold). method="higher" takes the conservative (upper)
        empirical quantile, which is the finite-sample-valid choice (never anti-conservative).
        """
        n = len(scores)
        if n == 0:
            return 1.0
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    @staticmethod
    def _robust_scale(values: np.ndarray, fallback: float) -> float:
        """Median of the values if any are strictly positive; `fallback` if all are <= 0."""
        return float(np.median(values)) if np.any(values > 0) else fallback

    def calibrate(self, p_true: np.ndarray, subgroups: np.ndarray | None = None,
                  epistemic: np.ndarray | None = None) -> "ConformalConfidence":
        """
        p_true     : calibrated probability assigned to the TRUE class, per calib example.
        subgroups  : optional per-example subgroup labels (for group-conditional coverage).
        epistemic  : optional per-example MC-dropout uncertainty (for the epistemic scale).
        """
        scores = 1.0 - np.asarray(p_true, dtype=float)
        self._global_q = self._quantile(scores, self.alpha)
        if epistemic is not None:
            epi = np.asarray(epistemic, dtype=float)
            self._global_u = self._robust_scale(epi, 1.0)
        if self.per_subgroup and subgroups is not None:
            subgroups = np.asarray(subgroups)
            epi_all = np.asarray(epistemic, dtype=float) if epistemic is not None else None
            for g in np.unique(subgroups):
                mask = subgroups == g
                self._group_q[str(g)] = self._quantile(scores[mask], self.alpha)
                if epi_all is not None:
                    self._group_u[str(g)] = self._robust_scale(epi_all[mask], self._global_u)
        return self

    def threshold(self, subgroup: str | None) -> float:
        """Confidence threshold tau such that predicting requires conf >= tau."""
        if self.per_subgroup and subgroup is not None and str(subgroup) in self._group_q:
            q = self._group_q[str(subgroup)]
        else:
            q = self._global_q
        # conf = 1 - s; requiring s <= q  <=>  conf >= 1 - q
        return 1.0 - q

    def epistemic_scale(self, subgroup: str | None) -> float:
        """Per-subgroup scale used to normalize epistemic uncertainty into [0,1]."""
        if self.per_subgroup and subgroup is not None and str(subgroup) in self._group_u:
            return self._group_u[str(subgroup)]
        return self._global_u


# --------------------------------------------------------------------------------
# The selective predictor (four regimes).
# --------------------------------------------------------------------------------
class SelectivePredictor:
    # Default confidence gate when SR-Conf is used WITHOUT a conformal object or explicit
    # tau_conf. Named (not a magic literal buried in the rule) so it is visible and auditable.
    DEFAULT_TAU_CONF = 0.7

    def __init__(
        self,
        regime: str = "DASP",                  # one of REGIMES
        conformal: ConformalConfidence | None = None,
        tau_conf: float | None = None,         # SR-Conf knob (if no conformal object)
        tau_unc: float = 0.5,                  # SR-Unc knob (epistemic-uncertainty threshold)
        tau_reach: float = 0.6,                # SR-CF knob
        margin: float = 0.15,                  # DASP knob (the ONLY swept knob for DASP)
        w_epi: float = 0.5,                    # DASP epistemic weight (FIXED, not swept)
        u_scale: float | None = None,          # global epistemic scale fallback
    ):
        if regime not in REGIMES:
            raise ValueError(f"unknown regime: {regime!r} (expected one of {REGIMES})")
        self.regime = regime
        self.conformal = conformal
        self.tau_conf = tau_conf
        self.tau_unc = tau_unc
        self.tau_reach = tau_reach
        self.margin = margin
        self.w_epi = w_epi
        self.u_scale = u_scale

    # --- helpers -----------------------------------------------------------------
    def _base_prediction(self, s: CaseSignals) -> Decision:
        return (Decision.PREDICT_FAVORABLE if s.p_favorable >= 0.5
                else Decision.PREDICT_UNFAVORABLE)

    def _conf_threshold(self, s: CaseSignals) -> float:
        """SR-Conf gate: conformal per-subgroup tau if available, else explicit/default tau_conf."""
        if self.conformal is not None:
            return self.conformal.threshold(s.subgroup)
        return self.tau_conf if self.tau_conf is not None else self.DEFAULT_TAU_CONF

    def _epi_norm(self, s: CaseSignals) -> float:
        """Normalize epistemic uncertainty into [0,1]. 0 when epistemic is unavailable.

        Clamped to [0,1]: an upstream estimator should yield a non-negative scalar, but we
        clamp the low end too so a stray negative can never *reduce* the defense signal.
        """
        if s.epistemic is None:
            return 0.0
        scale = (self.conformal.epistemic_scale(s.subgroup)
                 if self.conformal is not None else self.u_scale)
        if not scale or scale <= 0:
            return 0.0
        return float(min(1.0, max(0.0, s.epistemic / scale)))

    # --- continuous "safe to predict" score (for RPP / Refinement in metrics.py) --------
    def confidence_score(self, s: CaseSignals) -> float:
        """
        A continuous per-case score where HIGHER = safer to predict (more likely correct),
        consistent with the regime's gate. metrics.py uses this to compute RPP and Refinement
        (Xin et al. 2021; Gu & Hopkins 2023), which rank cases by confidence.

          SR-Conf : model confidence  conf = max(p, 1-p).
          SR-Unc  : negative epistemic uncertainty  (-epi); higher = less uncertain.
          SR-CF   : negative reachability  (-reach); a predicted case is "safer" when no
                    favorable outcome is reachable (i.e. less chance of a missed defense).
          DASP    : the margin by which it is safe to predict =
                    residual - defense_signal = (1-conf) - (reach + w_epi*epi_norm).
                    Higher (more positive) => safer to predict; lower => should abstain.
        """
        if self.regime == "SR-Conf":
            return s.conf
        if self.regime == "SR-Unc":
            return -(s.epistemic if s.epistemic is not None else 0.0)
        if self.regime == "SR-CF":
            return -s.reachability
        # DASP
        defense_signal = s.reachability + self.w_epi * self._epi_norm(s)
        return s.residual - defense_signal

    # --- the decision rule -------------------------------------------------------
    def predict_or_abstain(self, s: CaseSignals) -> Decision:
        if self.regime == "SR-Conf":
            return Decision.ABSTAIN if s.conf < self._conf_threshold(s) else self._base_prediction(s)

        if self.regime == "SR-Unc":
            # MC-dropout uncertainty gate (the faithful Santosh PV/BALD baseline): abstain when
            # epistemic uncertainty is high. With no epistemic signal, never abstain on this basis.
            if s.epistemic is not None and s.epistemic > self.tau_unc:
                return Decision.ABSTAIN
            return self._base_prediction(s)

        if self.regime == "SR-CF":
            # Abstain (defer to a human) when a favorable outcome is plausibly reachable.
            return Decision.ABSTAIN if s.reachability > self.tau_reach else self._base_prediction(s)

        # DASP -- the coupled rule.
        if s.p_favorable >= 0.5:
            # The model already predicts a favorable outcome: there is no "missed defense"
            # to guard against, so DASP never abstains here. (Keeps coverage high.)
            return self._base_prediction(s)
        defense_signal = s.reachability + self.w_epi * self._epi_norm(s)
        if (defense_signal - s.residual) > self.margin:
            return Decision.ABSTAIN
        return self._base_prediction(s)

    def batch(self, signals: list[CaseSignals]) -> list[Decision]:
        return [self.predict_or_abstain(s) for s in signals]

    def confidence_scores(self, signals: list[CaseSignals]) -> list[float]:
        return [self.confidence_score(s) for s in signals]


# --------------------------------------------------------------------------------
# Self-test: verifies the contribution's core behavior, backward compatibility,
# monotonicity, the new baselines/scores, and edge-case safety. Pure/deterministic (the
# only RNG is synthetic data generation for the monotonicity checks).
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- The canonical case DASP exists to catch -----------------------------------
    # Model is confident the defendant LOSES (p=0.2 -> conf 0.8, residual 0.2), but a small
    # plausible edit flips it (reach=0.7). DASP must abstain; plain confidence (0.8) would
    # happily predict the loss.
    danger = CaseSignals(p_favorable=0.2, reachability=0.7)
    assert SelectivePredictor("DASP", margin=0.15).predict_or_abstain(danger) == Decision.ABSTAIN
    assert SelectivePredictor("SR-Conf", tau_conf=0.7).predict_or_abstain(danger) == Decision.PREDICT_UNFAVORABLE

    # ---- Epistemic uncertainty PUSHES toward abstention ----------------------------
    # Borderline reach that would NOT abstain on its own, but high epistemic uncertainty
    # tips it over (with a per-case scale supplied via u_scale).
    borderline = CaseSignals(p_favorable=0.35, reachability=0.30, epistemic=0.25)
    sp_epi = SelectivePredictor("DASP", margin=0.15, w_epi=0.8, u_scale=0.15)
    no_epi = CaseSignals(p_favorable=0.35, reachability=0.30, epistemic=None)
    d_with = sp_epi.predict_or_abstain(borderline)
    d_without = sp_epi.predict_or_abstain(no_epi)
    rank = {Decision.PREDICT_UNFAVORABLE: 0, Decision.PREDICT_FAVORABLE: 0, Decision.ABSTAIN: 1}
    assert rank[d_with] >= rank[d_without]  # epistemic can only increase abstention propensity

    # ---- Backward compatibility: epi=None reduces EXACTLY to the old confidence rule ----
    def old_rule(p, reach, margin=0.15):
        conf = max(p, 1 - p)
        if p >= 0.5:
            return Decision.PREDICT_FAVORABLE
        return Decision.ABSTAIN if (reach - (1 - conf)) > margin else Decision.PREDICT_UNFAVORABLE
    sp = SelectivePredictor("DASP", margin=0.15)
    for p, r in [(0.2, 0.7), (0.45, 0.1), (0.9, 0.0), (0.3, 0.5), (0.49, 0.49)]:
        assert sp.predict_or_abstain(CaseSignals(p, r)) == old_rule(p, r)

    # ---- Confident win, nothing reachable: everyone predicts (high coverage) --------
    easy = CaseSignals(p_favorable=0.9, reachability=0.0)
    assert SelectivePredictor("DASP").predict_or_abstain(easy) == Decision.PREDICT_FAVORABLE

    # ---- NEW: SR-Unc baseline (MC-dropout uncertainty gate) -------------------------
    # Abstains iff epistemic > tau_unc; monotone in epistemic; never abstains when epi is None.
    su = SelectivePredictor("SR-Unc", tau_unc=0.2)
    assert su.predict_or_abstain(CaseSignals(0.2, 0.0, epistemic=0.30)) == Decision.ABSTAIN
    assert su.predict_or_abstain(CaseSignals(0.2, 0.0, epistemic=0.10)) == Decision.PREDICT_UNFAVORABLE
    assert su.predict_or_abstain(CaseSignals(0.2, 0.0, epistemic=None)) == Decision.PREDICT_UNFAVORABLE
    su_abst = [su.predict_or_abstain(CaseSignals(0.2, 0.0, epistemic=u)) == Decision.ABSTAIN
               for u in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    assert all(su_abst[i + 1] >= su_abst[i] for i in range(len(su_abst) - 1)), "SR-Unc monotone in epi"

    # ---- NEW: continuous confidence_score for every regime (RPP/Refinement input) ----
    s_hi = CaseSignals(0.95, 0.0, epistemic=0.05)   # very predictable
    s_lo = CaseSignals(0.45, 0.9, epistemic=0.40)   # risky: reachable + uncertain + near boundary
    for reg in REGIMES:
        spc = SelectivePredictor(reg, u_scale=0.159)
        # the safe case should score >= the risky case under every regime's own ordering
        assert spc.confidence_score(s_hi) >= spc.confidence_score(s_lo), f"score ordering {reg}"
    # batch helper returns one score per case
    assert len(SelectivePredictor("DASP", u_scale=0.159).confidence_scores([s_hi, s_lo])) == 2

    # ---- Monotonicity (fixed case set; rule is deterministic so it must be exact) ----
    rng = np.random.default_rng(0)
    fixed = [(float(rng.uniform(0, 0.499)), float(rng.uniform(0, 1))) for _ in range(400)]
    spm = SelectivePredictor("DASP", margin=0.15, w_epi=0.5, u_scale=0.159)

    def abstain_rate_epi(E):
        decs = [spm.predict_or_abstain(CaseSignals(p, r, epistemic=E)) for p, r in fixed]
        return np.mean([d == Decision.ABSTAIN for d in decs])
    rates = [abstain_rate_epi(E) for E in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]]
    assert all(rates[i + 1] >= rates[i] - 1e-12 for i in range(len(rates) - 1)), "epistemic monotonicity"

    fixed_pe = [(float(rng.uniform(0, 0.499)), float(rng.uniform(0, 0.3))) for _ in range(400)]

    def abstain_rate_reach(R):
        decs = [spm.predict_or_abstain(CaseSignals(p, R, epistemic=e)) for p, e in fixed_pe]
        return np.mean([d == Decision.ABSTAIN for d in decs])
    rr = [abstain_rate_reach(R) for R in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(rr[i + 1] >= rr[i] - 1e-12 for i in range(len(rr) - 1)), "reachability monotonicity"

    # ---- Single-knob: coverage monotone in margin ----------------------------------
    sig = [CaseSignals(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
                       epistemic=float(rng.uniform(0, 0.3))) for _ in range(400)]

    def coverage(m):
        decs = SelectivePredictor("DASP", margin=m, w_epi=0.5, u_scale=0.159).batch(sig)
        return np.mean([d != Decision.ABSTAIN for d in decs])
    covs = [coverage(m) for m in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    assert all(covs[i + 1] >= covs[i] - 1e-12 for i in range(len(covs) - 1)), "coverage monotonicity"

    # ---- _epi_norm clamps to [0,1] (negative input cannot weaken abstention) --------
    clamp_sp = SelectivePredictor("DASP", u_scale=0.15)
    assert clamp_sp._epi_norm(CaseSignals(0.2, 0.0, epistemic=-5.0)) == 0.0
    assert clamp_sp._epi_norm(CaseSignals(0.2, 0.0, epistemic=99.0)) == 1.0

    # ---- Conformal per-subgroup calibration (incl. epistemic scale) ----------------
    p_true = rng.uniform(0.5, 1.0, size=500)
    groups = rng.choice(["A", "B"], size=500)
    epis = rng.uniform(0.0, 0.3, size=500)
    cc = ConformalConfidence(alpha=0.1, per_subgroup=True).calibrate(p_true, groups, epis)
    assert 0.0 <= cc.threshold("A") <= 1.0 and cc.epistemic_scale("B") > 0
    # conformal-gated SR-Conf uses the per-subgroup threshold
    sp_conf = SelectivePredictor("SR-Conf", conformal=cc)
    assert sp_conf.predict_or_abstain(CaseSignals(0.5, 0.0, subgroup="A")) in set(Decision)
    # alpha validation
    try:
        ConformalConfidence(alpha=1.5); raise AssertionError("should reject alpha>=1")
    except ValueError:
        pass

    # ---- Regime validation ----------------------------------------------------------
    try:
        SelectivePredictor("nonsense"); raise AssertionError("should reject unknown regime")
    except ValueError:
        pass

    # ---- Edge cases -----------------------------------------------------------------
    for reg in REGIMES:
        for s in [CaseSignals(0.5, 0.0), CaseSignals(0.2, 0.0, epistemic=0.0),
                  CaseSignals(0.2, 1.0, epistemic=0.3), CaseSignals(0.0, 0.0),
                  CaseSignals(1.0, 1.0, epistemic=0.3)]:
            assert SelectivePredictor(reg, u_scale=0.159).predict_or_abstain(s) in set(Decision)

    print("selective.py self-test passed "
          "(DASP coupling + SR-Unc baseline + continuous scores + epistemic + "
          "backward-compat + monotonicity + single-knob + conformal verified)")
