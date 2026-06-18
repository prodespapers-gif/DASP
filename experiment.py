"""
experiment.py
================================================================================
End-to-end orchestration for the Defense-Aware Selective Prediction (DASP) system.
Single entry point invoked by scripts/run_all.sh. Reads configs/default.yaml.

This file contains ORCHESTRATION ONLY; every piece of method logic lives in a
dedicated module (data, model, plausibility, counterfactual, selective, metrics) so
the scientific contribution remains auditable in isolation.

Pipeline (each stage cached/checkpointed so partial reruns are cheap):

  1. data       load ILDC (primary) or ECtHR (secondary)            [data.py]
  2. predict    fit InLegalBERT (or TF-IDF baseline) WITH a dev set  [model.py]
                so temperature scaling produces CALIBRATED probs
  3. uncertain  Monte-Carlo-dropout epistemic uncertainty on test    [model.py]
                (inlegalbert only; cached to disk; TF-IDF -> None)
  4. calibrate  split conformal, per-subgroup, on dev, INCLUDING the  [selective.py]
                epistemic scale used by DASP
  5. cf         Polyjuice generation + legal-plausibility filter ->   [counterfactual.py
                counterfactual reachability per case (cached)          + plausibility.py]
  6. selective  build CaseSignals(p, reach, subgroup, EPISTEMIC);     [selective.py]
                run SR-Conf / SR-CF / DASP with the conformal object
  7. metrics    Defense Risk-Coverage curves, subgroup coverage,      [metrics.py]
                actionable coverage, CF quality
  8. ablate     DASP margin sensitivity; epistemic-on vs -off; etc.
  9. persist    write results/results_<dataset>_seed<k>.json

Multi-seed: pass --seed to override the config seed; each seed writes its own results
file. Run one seed per GPU (see scripts/run_all.sh). Then `--aggregate` combines all
per-seed files into mean +/- std, bootstrap confidence intervals on the DRC curves, and
a paired test of DASP vs. the baselines.

KEY CORRECTNESS POINTS:
  * The predictor is trained WITH dev_texts/dev_labels so temperature scaling runs;
    without it the probabilities are uncalibrated and every threshold is meaningless.
  * Epistemic uncertainty (MC-dropout) is computed and threaded into BOTH the conformal
    calibration (per-subgroup scale) and the CaseSignals, so the DASP coupling is active.
  * The SelectivePredictor receives the conformal object, so it normalizes epistemic
    per-subgroup and the fairness guarantee covers both signals.
  * Graceful degradation: with the TF-IDF baseline (or use_mc_dropout=false), epistemic
    is None and DASP reduces to the confidence-only coupling -- the pipeline still runs.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

import counterfactual as CF
import data as D
import metrics as ME
import model as M
import plausibility as PL
import selective as SEL


# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def make_model_config(cfg: dict, seed: int) -> "M.ModelConfig":
    """Map the YAML model.inlegalbert block into a typed ModelConfig.

    Unknown keys are not accepted by ModelConfig, so the block is filtered to the
    dataclass's fields. This means any field the YAML omits (e.g. epistemic_estimator)
    takes the ModelConfig default; to drive the std/PV/BALD ablation, set
    model.inlegalbert.epistemic_estimator in the YAML and it flows through here.
    """
    block = dict(cfg["model"].get("inlegalbert", {}))
    block["seed"] = seed
    # device/uncertainty knobs may live at the top of the model block
    for k in ("device",):
        if k in cfg["model"]:
            block[k] = cfg["model"][k]
    import dataclasses
    valid = {f.name for f in dataclasses.fields(M.ModelConfig)}
    filtered = {k: v for k, v in block.items() if k in valid}
    return M.ModelConfig(**filtered)


# --------------------------------------------------------------------------------
# Stage 2: predictor (trained WITH a dev set so temperature scaling runs)
# --------------------------------------------------------------------------------
def stage_predict(cfg: dict, splits: dict, seed: int):
    kind = cfg["model"]["kind"]
    if kind == "inlegalbert":
        predictor = M.InLegalBERTClassifier(make_model_config(cfg, seed))
    elif kind == "tfidf":
        predictor = M.TfidfLinearBaseline(seed=seed)
    else:
        raise ValueError(f"unknown model.kind: {kind!r}")

    tr, dv = splits["train"], splits["dev"]
    # Pass dev so InLegalBERT fits a temperature (TF-IDF ignores the extra args).
    predictor.fit([e.text for e in tr], [e.label for e in tr],
                  [e.text for e in dv], [e.label for e in dv])
    return predictor


def batched_proba(predictor, texts, batch_size: int = 32) -> np.ndarray:
    """Batched deterministic P(y=1) over many docs."""
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(np.asarray(predictor.predict_proba(texts[i:i + batch_size])))
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------------
# Stage 3: epistemic uncertainty (MC-dropout), cached
# --------------------------------------------------------------------------------
def stage_uncertainty(cfg: dict, predictor, texts, cache_path: str) -> np.ndarray | None:
    """
    Returns per-doc epistemic std (MC-dropout), or None if unavailable/disabled.
    Cached to disk so repeated evaluation runs do not recompute the T*N passes.
    """
    if cfg["model"]["kind"] != "inlegalbert" or not cfg["selective"].get("use_mc_dropout", True):
        return None
    if not hasattr(predictor, "epistemic_uncertainty"):
        return None
    if os.path.exists(cache_path):
        return np.load(cache_path)
    epi = predictor.epistemic_uncertainty(texts)  # batched internally in model.py
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, epi)
    return epi


# --------------------------------------------------------------------------------
# Stage 4: conformal calibration WITH per-subgroup epistemic scale
# --------------------------------------------------------------------------------
def stage_calibrate(cfg: dict, predictor, splits: dict, dataset: str = ""):
    dev = splits["dev"]
    p = batched_proba(predictor, [e.text for e in dev])
    p_true = np.where(np.array([e.label for e in dev]) == 1, p, 1 - p)
    groups = np.array([e.subgroup for e in dev])
    # epistemic on dev (for the per-subgroup scale). The cache key includes the dataset so
    # ILDC and ECtHR (different dev sets) never share a dev-epistemic cache file.
    dev_epi = stage_uncertainty(
        cfg, predictor, [e.text for e in dev],
        cache_path=_cache_name(cfg, "dev_epi", dataset))
    return SEL.ConformalConfidence(
        alpha=cfg["selective"]["alpha"],
        per_subgroup=cfg["selective"]["per_subgroup"],
    ).calibrate(p_true, groups, epistemic=dev_epi)


# --------------------------------------------------------------------------------
# Stage 5: counterfactual reachability (cached, versioned, control-code steered)
# --------------------------------------------------------------------------------
def stage_counterfactual(cfg: dict, splits: dict, predictor, dataset: str = ""):
    generate_fn = (_make_polyjuice(cfg) if cfg["cf"]["use_polyjuice"]
                   else _make_noop_generator())

    # learned plausibility tier is optional; rules-only by default for speed
    scorer = None
    if cfg["cf"].get("use_learned_plausibility", False) and cfg["model"]["kind"] == "inlegalbert":
        scorer = M.PlausibilityScorer(seed=cfg["seed"], device=cfg["model"].get("device", "cuda"))
        # NOTE: scorer.fit(pairs, labels) must be called on the annotated edit set before use;
        # see the annotation protocol. Left unfit here -> returns neutral 0.5 (rules tier governs).

    filt = PL.LegalPlausibilityFilter(
        scorer=scorer,
        threshold=cfg["cf"]["plausibility_threshold"],
        rules_only=cfg["cf"]["rules_only"],
    )

    def predict_fn(text: str) -> float:
        return float(predictor.predict_proba([text])[0])

    eng = CF.CounterfactualEngine(
        generate_fn, predict_fn, filt,
        cache_path=cfg["paths"]["cf_cache"], flip_threshold=0.5,
        cache_version=_cf_cache_version(cfg, dataset))
    return eng.run(splits["test"], resume=True)


def _cf_cache_version(cfg: dict, dataset: str) -> str:
    """
    Bind the counterfactual cache to the model + generator + dataset, so a model or
    generator change invalidates stale counterfactuals instead of silently reusing them.
    """
    import hashlib
    model_tag = cfg["model"].get("inlegalbert", {}).get("model_tag", cfg["model"]["kind"])
    gen = "polyjuice" if cfg["cf"]["use_polyjuice"] else "noop"
    seed = cfg.get("_seed", cfg.get("seed", 0))
    key = f"{model_tag}|{gen}|{dataset}|seed{seed}|n{cfg['cf'].get('num_perturbations', 0)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------
# Stage 6: build CaseSignals with the epistemic signal
# --------------------------------------------------------------------------------
def stage_signals(predictor, splits: dict, cf_results: dict,
                  epistemic: np.ndarray | None) -> list[SEL.CaseSignals]:
    test = splits["test"]
    p = batched_proba(predictor, [e.text for e in test])
    sig = []
    for i, e in enumerate(test):
        r = cf_results.get(e.id)
        reach = r.reachability if r else 0.0
        epi = float(epistemic[i]) if epistemic is not None else None
        sig.append(SEL.CaseSignals(p_favorable=float(p[i]), reachability=reach,
                                   subgroup=e.subgroup, epistemic=epi))
    return sig


# --------------------------------------------------------------------------------
# Stage 7: evaluation -- DRC curves (all regimes incl. SR-Unc) + RPP/Refinement
# --------------------------------------------------------------------------------
def stage_evaluate(cfg: dict, signals, conformal, y_true: list[int] | None = None) -> dict:
    out = {"regimes": {}, "subgroup": {}, "actionable_coverage": None, "selective_metrics": {}}
    pts = cfg["eval"]["sweep_points"]
    w_epi = cfg["selective"].get("w_epi", 0.5)

    sweeps = {
        "SR-Conf": np.linspace(0.5, 0.99, pts),
        "SR-CF": np.linspace(0.1, 0.95, pts),
        "DASP": np.linspace(0.0, 0.5, pts),
    }
    # SR-Unc (the Santosh MC-uncertainty baseline) is only meaningful when an epistemic
    # signal exists; sweep tau_unc over the observed epistemic range.
    epi_vals = [s.epistemic for s in signals if s.epistemic is not None]
    if epi_vals:
        hi = max(epi_vals)
        sweeps["SR-Unc"] = np.linspace(0.0, hi if hi > 0 else 1.0, pts)

    for regime, sw in sweeps.items():
        # metrics.defense_risk_coverage builds its own SelectivePredictor per threshold;
        # for DASP we must give it the conformal object + w_epi so epistemic is normalized.
        extra = {"conformal": conformal, "w_epi": w_epi} if regime == "DASP" else {}
        out["regimes"][regime] = ME.defense_risk_coverage(
            signals, regime, sweep=sw,
            reach_threshold=cfg["eval"]["reach_threshold"], **extra)

    sp = SEL.SelectivePredictor("DASP", conformal=conformal,
                                margin=cfg["selective"]["dasp_margin"], w_epi=w_epi)
    out["subgroup"] = ME.subgroup_coverage(signals, sp, cfg["eval"]["reach_threshold"])
    out["actionable_coverage"] = ME.actionable_coverage(
        signals, conf_threshold=cfg["eval"]["actionable_conf"])

    # Standard selective-prediction metrics (RPP, Refinement; Santosh Eq. 4-5) per regime,
    # at the operating point used for subgroup analysis. Requires per-case correctness.
    if y_true is not None:
        for regime in out["regimes"]:
            kw = {"conformal": conformal, "w_epi": w_epi,
                  "margin": cfg["selective"]["dasp_margin"]} if regime == "DASP" else {}
            out["selective_metrics"][regime] = ME.selective_metrics(
                signals, regime, y_true, **kw)
    return out


# --------------------------------------------------------------------------------
# Stage 8: ablations -- margin sensitivity, epistemic on/off, std/PV/BALD estimator
# --------------------------------------------------------------------------------
def stage_ablations(cfg: dict, signals, conformal) -> dict:
    w_epi = cfg["selective"].get("w_epi", 0.5)
    abl = {"dasp_margin_sensitivity": {}, "epistemic_ablation": {}}

    for margin in cfg["eval"]["ablation_margins"]:
        sp = SEL.SelectivePredictor("DASP", conformal=conformal, margin=margin, w_epi=w_epi)
        decs = sp.batch(signals)
        cov = sum(d != SEL.Decision.ABSTAIN for d in decs) / len(signals)
        abl["dasp_margin_sensitivity"][f"{margin:.2f}"] = {"coverage": cov}

    # Epistemic ablation: same signals, DASP with epistemic ON vs OFF (w_epi=0).
    # Demonstrates the contribution of the MC-dropout coupling specifically.
    for label, we in (("epistemic_on", w_epi), ("epistemic_off", 0.0)):
        drc = ME.defense_risk_coverage(
            signals, "DASP", sweep=np.linspace(0.0, 0.5, cfg["eval"]["sweep_points"]),
            reach_threshold=cfg["eval"]["reach_threshold"], conformal=conformal, w_epi=we)
        abl["epistemic_ablation"][label] = {"aurc": drc["aurc"]}
    return abl


def stage_estimator_ablation(cfg: dict, predictor, splits: dict, cf_results: dict,
                             conformal, dataset: str) -> dict:
    """
    Ablate the epistemic estimator (predictive std vs PV vs BALD; Santosh Eq. 7-9) on the
    SAME MC-dropout samples, reporting DASP AURC for each. Only runs when the predictor
    exposes MC sampling (InLegalBERT); returns {} for the TF-IDF baseline. The MC samples
    are computed once and cached, so the three estimators share one forward-pass budget.
    """
    if cfg["model"]["kind"] != "inlegalbert" or not hasattr(predictor, "predict_proba_mc"):
        return {}
    if not cfg["selective"].get("use_mc_dropout", True):
        return {}

    test = splits["test"]
    cache = _cache_name(cfg, "test_mc_samples", dataset)
    if os.path.exists(cache):
        samples = np.load(cache)
    else:
        samples = predictor.predict_proba_mc([e.text for e in test])  # (N, T)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, samples)

    p = batched_proba(predictor, [e.text for e in test])
    w_epi = cfg["selective"].get("w_epi", 0.5)
    out = {}
    for kind in ("std", "pv", "bald"):
        u = M.uncertainty_from_samples(samples, kind)
        sig = []
        for i, e in enumerate(test):
            r = cf_results.get(e.id)
            sig.append(SEL.CaseSignals(
                p_favorable=float(p[i]), reachability=(r.reachability if r else 0.0),
                subgroup=e.subgroup, epistemic=float(u[i])))
        drc = ME.defense_risk_coverage(
            sig, "DASP", sweep=np.linspace(0.0, 0.5, cfg["eval"]["sweep_points"]),
            reach_threshold=cfg["eval"]["reach_threshold"], conformal=conformal, w_epi=w_epi)
        out[kind] = {"aurc": drc["aurc"]}
    return out


# --------------------------------------------------------------------------------
# Counterfactual generator factories (lazy; heavy)
# --------------------------------------------------------------------------------
def _make_polyjuice(cfg: dict):
    # Control codes that realize the mutable legal edit types (the legal<->linguistic
    # bridge from plausibility.py). Steering Polyjuice toward these raises the yield of
    # edits the plausibility filter will accept, instead of generating freely and
    # discarding most candidates.
    steer_codes = sorted({c for et in PL.EDIT_TYPE_TO_POLYJUICE
                          for c in PL.EDIT_TYPE_TO_POLYJUICE[et]})

    def generate_fn(text: str):
        from polyjuice import Polyjuice
        if generate_fn._pj is None:
            is_cuda = cfg["model"].get("device", "cuda") == "cuda"
            generate_fn._pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=is_cuda)
        n = cfg["cf"]["num_perturbations"]
        # Request targeted edits per control code, falling back to free generation if this
        # Polyjuice build does not accept a control code argument.
        perturbations: list[str] = []
        try:
            per_code = max(1, n // max(1, len(steer_codes)))
            for code in steer_codes:
                perturbations += generate_fn._pj.perturb(
                    text, ctrl_code=code, num_perturbations=per_code)
        except TypeError:
            perturbations = generate_fn._pj.perturb(text, num_perturbations=n)
        # de-duplicate while preserving order, then attach the changed-span offsets
        seen, uniq = set(), []
        for ed in perturbations:
            if ed not in seen:
                seen.add(ed)
                uniq.append(ed)
        return [(ed, *_first_diff_span(text, ed)) for ed in uniq]
    generate_fn._pj = None
    return generate_fn


def _make_noop_generator():
    def generate_fn(text):
        return []
    return generate_fn


def _first_diff_span(a: str, b: str):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return i, max(i, len(a) - j)


# --------------------------------------------------------------------------------
# Multi-seed aggregation: mean/std, bootstrap CI on DRC, paired test
# --------------------------------------------------------------------------------
def aggregate(cfg: dict, dataset: str) -> dict:
    """Combine all results_<dataset>_seed*.json into summary statistics."""
    results_dir = cfg["paths"]["results"]
    files = sorted(f for f in os.listdir(results_dir)
                   if f.startswith(f"results_{dataset}_seed") and f.endswith(".json"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 per-seed files to aggregate; found {len(files)} in {results_dir}")
    runs = [json.load(open(os.path.join(results_dir, f))) for f in files]

    # Common coverage grid on [0,1], matching metrics.py's AURC integration support so the
    # aggregated curve and the per-seed AURC are on the same axis.
    common_cov = np.linspace(0.0, 1.0, cfg["eval"]["sweep_points"])
    summary = {"n_seeds": len(runs), "datasets": dataset, "regimes": {}}

    # Regimes actually present across all runs (SR-Unc appears only when epistemic exists).
    regimes = [r for r in ("SR-Conf", "SR-Unc", "SR-CF", "DASP")
               if all(r in run["regimes"] for run in runs)]

    for regime in regimes:
        # per-seed AURC
        aurcs = np.array([r["regimes"][regime]["aurc"] for r in runs])
        # interpolate each seed's DRC onto the common coverage grid, then mean/std pointwise
        curves = []
        for r in runs:
            cov = np.asarray(r["regimes"][regime]["coverage"])
            risk = np.asarray(r["regimes"][regime]["defense_risk"])
            order = np.argsort(cov)
            curves.append(np.interp(common_cov, cov[order], risk[order]))
        curves = np.vstack(curves)
        # bootstrap CI on the mean curve (resample seeds)
        lo, hi = _bootstrap_curve_ci(curves, n_boot=cfg["eval"].get("n_bootstrap", 2000))
        summary["regimes"][regime] = {
            "aurc_mean": float(aurcs.mean()),
            "aurc_std": float(aurcs.std(ddof=1)) if len(aurcs) > 1 else 0.0,
            "drc_common_coverage": common_cov.tolist(),
            "drc_mean_risk": curves.mean(0).tolist(),
            "drc_std_risk": curves.std(0, ddof=1).tolist() if len(curves) > 1 else [0.0] * len(common_cov),
            "drc_ci_lo": lo.tolist(),
            "drc_ci_hi": hi.tolist(),
        }

    # paired test: DASP vs each available baseline on per-seed AURC (lower is better)
    summary["paired_tests"] = {}
    dasp = np.array([r["regimes"]["DASP"]["aurc"] for r in runs])
    for base in [b for b in ("SR-Conf", "SR-Unc", "SR-CF") if b in regimes]:
        other = np.array([r["regimes"][base]["aurc"] for r in runs])
        summary["paired_tests"][f"DASP_vs_{base}"] = _paired_test(dasp, other)
    return summary


def _bootstrap_curve_ci(curves: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    r = np.random.default_rng(seed)
    n = curves.shape[0]
    boot_means = np.empty((n_boot, curves.shape[1]))
    for b in range(n_boot):
        idx = r.integers(0, n, n)
        boot_means[b] = curves[idx].mean(0)
    lo = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank (paired, nonparametric). Honest small-n caveat included."""
    out = {"a_mean": float(a.mean()), "b_mean": float(b.mean()), "n": int(len(a))}
    diffs = np.asarray(a) - np.asarray(b)
    if np.allclose(diffs, 0.0):
        # all paired differences are zero -> the signed-rank statistic is undefined.
        out["note"] = "all paired differences are zero; signed-rank test not applicable"
        return out
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b)
        out["wilcoxon_stat"] = float(stat)
        out["p_value"] = float(p)
    except Exception as e:
        out["note"] = f"wilcoxon unavailable: {e}"
    if len(a) < 6:
        out["caveat"] = ("with n<6 seeds the minimum achievable Wilcoxon p-value is >0.05; "
                         "rely on the per-coverage bootstrap CIs for statistical power.")
    return out


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def run_single(cfg: dict, dataset: str, seed: int) -> str:
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    cfg = dict(cfg)
    cfg["_seed"] = seed  # used by _cache_name

    max_samples = cfg.get("smoke", {}).get("max_samples")
    if max_samples:
        print(f"[experiment] SMOKE: subsampling each split to {max_samples} (seeded, stratified)")
    splits = (D.load_ildc(cfg.get("hf_token"), cfg["paths"]["data_cache"],
                          max_samples=max_samples, seed=seed,
                          processed_cache=cfg["paths"].get("processed_cache"))
              if dataset == "ildc"
              else D.load_ecthr(cfg["paths"]["data_cache"], cfg["eval"]["ecthr_task"],
                               max_samples=max_samples, seed=seed,
                               processed_cache=cfg["paths"].get("processed_cache")))

    predictor = stage_predict(cfg, splits, seed)
    conformal = stage_calibrate(cfg, predictor, splits, dataset)
    test_epi = stage_uncertainty(cfg, predictor, [e.text for e in splits["test"]],
                                 cache_path=_cache_name(cfg, "test_epi", dataset))
    cf_results = stage_counterfactual(cfg, splits, predictor, dataset)
    signals = stage_signals(predictor, splits, cf_results, test_epi)

    y_true = [int(e.label) for e in splits["test"]]
    results = stage_evaluate(cfg, signals, conformal, y_true=y_true)
    results["ablations"] = stage_ablations(cfg, signals, conformal)
    results["ablations"]["estimator_ablation"] = stage_estimator_ablation(
        cfg, predictor, splits, cf_results, conformal, dataset)
    results["meta"] = {
        "dataset": dataset, "seed": seed, "n_test": len(splits["test"]),
        "data_hash": D.content_hash(splits["test"]),
        "model_kind": cfg["model"]["kind"],
        "epistemic_used": test_epi is not None,
        "epistemic_estimator": (cfg["model"].get("inlegalbert", {}).get("epistemic_estimator", "bald")
                                if test_epi is not None else None),
        "temperature": getattr(predictor, "temperature", None),
        "calibration": getattr(predictor, "calibration", None),  # {ece_before, ece_after, ...}
    }
    out_path = os.path.join(cfg["paths"]["results"], f"results_{dataset}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}  (epistemic_used={results['meta']['epistemic_used']})")
    return out_path


def _cache_name(cfg: dict, tag: str, dataset: str = "") -> str:
    seed = cfg.get("_seed", cfg.get("seed", 0))
    model_tag = cfg["model"].get("inlegalbert", {}).get("model_tag", cfg["model"]["kind"])
    base = cfg["paths"].get("feat_cache", "data/feat_cache")
    fname = f"{tag}_{dataset}_{model_tag}_seed{seed}.npy".replace("__", "_")
    return os.path.join(base, fname)


def main():
    ap = argparse.ArgumentParser(description="DASP experiment orchestrator")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", choices=["ildc", "ecthr"], default="ildc")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (run one per GPU for multi-seed)")
    ap.add_argument("--aggregate", action="store_true",
                    help="combine all per-seed result files into summary statistics")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.aggregate:
        summary = aggregate(cfg, args.dataset)
        out = os.path.join(cfg["paths"]["results"], f"summary_{args.dataset}.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out}  ({summary['n_seeds']} seeds aggregated)")
        return

    seed = args.seed if args.seed is not None else cfg["seed"]
    run_single(cfg, args.dataset, seed)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        # Lightweight end-to-end smoke test on synthetic data (no torch, no network, no
        # gated download). Asserts the pipeline produces every expected results key, so
        # wiring regressions are caught without the real datasets.
        import random
        import tempfile

        def _synth(n, seed):
            rng = random.Random(seed)
            out = []
            for i in range(n):
                fav = rng.random() < 0.5
                base = ("appeal allowed conviction set aside witness " if fav
                        else "appeal dismissed conviction upheld evidence ")
                out.append(D.Example(
                    id=f"{1980 + (i % 40)}_{i}", text=base * 3 + f" PW-{i % 5} 5th May 2001",
                    label=int(fav), subgroup=D._subgroup_of({"id": f"{1980 + (i % 40)}_{i}"})))
            return out

        D.load_ildc = lambda token, cache, **kw: {
            "train": _synth(120, 1), "dev": _synth(60, 2),
            "test": _synth(80, 3), "expert": _synth(8, 4)}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "seed": 1, "model": {"kind": "tfidf"},
                "selective": {"alpha": 0.1, "per_subgroup": True, "use_mc_dropout": True,
                              "w_epi": 0.5, "dasp_margin": 0.15},
                "cf": {"use_polyjuice": False, "rules_only": True,
                       "plausibility_threshold": 0.5, "num_perturbations": 4},
                "eval": {"sweep_points": 11, "reach_threshold": 0.6, "actionable_conf": 0.7,
                         "ablation_margins": [0.1, 0.2, 0.3], "ecthr_task": "ecthr_a"},
                "paths": {"results": os.path.join(tmp, "res"),
                          "data_cache": os.path.join(tmp, "data"),
                          "cf_cache": os.path.join(tmp, "cf", "cf.json"),
                          "feat_cache": os.path.join(tmp, "feat")},
            }
            os.makedirs(cfg["paths"]["results"], exist_ok=True)
            path = run_single(cfg, "ildc", seed=1)
            res = json.load(open(path))
            # the three baselines + DASP are all present (SR-Unc only with epistemic; TF-IDF -> none)
            assert {"SR-Conf", "SR-CF", "DASP"} <= set(res["regimes"])
            # standard selective metrics computed per regime
            assert "selective_metrics" in res and "DASP" in res["selective_metrics"]
            assert {"rpp", "refinement"} <= set(res["selective_metrics"]["DASP"])
            # ablations present (estimator ablation empty for TF-IDF, but key exists)
            assert {"dasp_margin_sensitivity", "epistemic_ablation",
                    "estimator_ablation"} <= set(res["ablations"])
            # meta records calibration slot + estimator
            assert "calibration" in res["meta"] and "epistemic_estimator" in res["meta"]
            print("experiment.py self-test passed (pipeline keys, selective metrics, ablations, meta)")
        sys.exit(0)
    main()
