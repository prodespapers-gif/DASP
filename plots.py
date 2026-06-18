"""
plots.py
================================================================================
Generate every camera-ready figure and table for the paper from the JSON produced
by experiment.py. Reads TWO kinds of file:

  * summary_<dataset>.json    (from `experiment.py --aggregate`): cross-seed mean DRC
                              curve + bootstrap confidence band + AURC mean/std + paired
                              tests.  -> the HEADLINE Figure 1.
  * results_<dataset>_seed*.json  (per-seed): subgroup coverage, margin sensitivity, the
                              epistemic on/off ablation, actionable coverage.  Aggregated
                              ACROSS seeds here (mean +/- std) so the supporting figures are
                              statistically consistent with the headline figure.

Outputs (for every dataset found):
  Fig 1  DRC_<dataset>          mean Defense Risk-Coverage of SR-Conf / SR-Unc / SR-CF /
                                DASP with a bootstrap CI band; DASP visually dominant.
  Fig 2  EPISTEMIC_<dataset>    epistemic-ON vs -OFF AURC (isolates the MC-dropout coupling
                                that is the contribution).
  Fig 3  SUBGROUP_<dataset>     subgroup-conditional coverage & defense-risk (fairness),
                                mean +/- std across seeds.
  Fig 4  MARGIN_<dataset>       DASP margin sensitivity (coverage vs margin), mean +/- std.
  Fig 5  CALIBRATION_<dataset>  Expected Calibration Error before vs after temperature
                                scaling (Guo 2017); the standard calibration evidence.
  Fig 6  ESTIMATOR_<dataset>    DASP AURC under std / PV / BALD epistemic estimators
                                (Santosh 2024); supports the choice of BALD.
  Tab 1  table_headline.tex     AURC (mean +/- std) per regime per dataset + actionable
                                coverage + paired-test p-values.  (LaTeX, booktabs.)
  Tab 2  table_epistemic.tex    epistemic on/off AURC per dataset.  (LaTeX.)
  Tab 3  table_selective.tex    RPP & Refinement per regime (standard selective-prediction
                                metrics; Santosh Eq. 4-5).  (LaTeX.)
  manifest.json                 provenance: which JSON files produced each figure.

Figures whose underlying data is absent (e.g. calibration/estimator on a TF-IDF run, or
SR-Unc when there is no epistemic signal) are gracefully skipped rather than emitted empty.

Design choices for publication quality:
  * Colorblind-safe palette (Wong 2011) AND distinct linestyles/markers per regime, so
    figures are legible without color (Elsevier accessibility guidance).
  * DASP is solid + thick + saturated (it is the contribution); baselines are muted with
    dashed/dotted lines.
  * Vector PDF (for LaTeX \\includegraphics) AND raster PNG (for quick preview) are both
    written for every figure.
  * Serif fonts and sizes tuned for single-column Elsevier width.

Pure matplotlib (no seaborn). CPU-trivial. No network.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------------
# Publication style
# --------------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette.
CB = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

# Per-regime visual encoding: DASP dominant (solid, thick, saturated); baselines muted.
REGIME_STYLE = {
    "DASP":    {"color": CB["vermillion"], "ls": "-",  "lw": 2.6, "marker": "o", "ms": 4, "z": 4},
    "SR-Conf": {"color": CB["blue"],       "ls": "--", "lw": 1.8, "marker": "s", "ms": 3, "z": 2},
    "SR-Unc":  {"color": CB["purple"],     "ls": "-.", "lw": 1.8, "marker": "D", "ms": 3, "z": 2},
    "SR-CF":   {"color": CB["green"],      "ls": ":",  "lw": 1.8, "marker": "^", "ms": 3, "z": 2},
}

# Canonical regime order (baselines first, DASP last so it draws on top of the others).
REGIME_ORDER = ("SR-Conf", "SR-Unc", "SR-CF", "DASP")
BASELINES = ("SR-Conf", "SR-Unc", "SR-CF")


def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "lines.antialiased": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in vector output (Elsevier-friendly)
        "ps.fonttype": 42,
    })


def save_fig(fig, fig_dir: str, name: str) -> list[str]:
    """Write both vector PDF and raster PNG; return the paths."""
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def discover(results_dir: str):
    """Return {dataset: {'summary': path|None, 'seeds': [paths]}}."""
    out = defaultdict(lambda: {"summary": None, "seeds": []})
    for fn in sorted(os.listdir(results_dir)):
        m_sum = re.fullmatch(r"summary_(\w+)\.json", fn)
        m_seed = re.fullmatch(r"results_(\w+)_seed(\d+)\.json", fn)
        if m_sum:
            out[m_sum.group(1)]["summary"] = os.path.join(results_dir, fn)
        elif m_seed:
            out[m_seed.group(1)]["seeds"].append(os.path.join(results_dir, fn))
    return out


def aggregate_seeds(seed_paths: list[str]) -> dict:
    """
    Aggregate the per-seed supporting metrics (subgroup, margin, epistemic ablation,
    actionable coverage) across seeds into mean +/- std. Robust to a single seed.
    """
    runs = [load_json(p) for p in seed_paths]
    n = len(runs)
    agg = {"n_seeds": n}
    if n == 0:
        return agg  # summary-only dataset; supporting figures are skipped downstream

    # actionable coverage
    ac = np.array([r.get("actionable_coverage", np.nan) for r in runs], dtype=float)
    agg["actionable_coverage"] = {"mean": float(np.nanmean(ac)),
                                  "std": float(np.nanstd(ac, ddof=1)) if n > 1 else 0.0}

    # subgroup coverage + defense risk
    groups = list(runs[0].get("subgroup", {}).keys()) if runs else []
    sg = {}
    for g in groups:
        cov = np.array([r["subgroup"][g]["coverage"] for r in runs if g in r.get("subgroup", {})])
        risk = np.array([r["subgroup"][g]["defense_risk"] for r in runs if g in r.get("subgroup", {})])
        sg[g] = {
            "cov_mean": float(cov.mean()), "cov_std": float(cov.std(ddof=1)) if len(cov) > 1 else 0.0,
            "risk_mean": float(risk.mean()), "risk_std": float(risk.std(ddof=1)) if len(risk) > 1 else 0.0,
        }
    agg["subgroup"] = sg

    # DASP margin sensitivity (coverage vs margin)
    margins = sorted(runs[0].get("ablations", {}).get("dasp_margin_sensitivity", {}).keys(),
                     key=float) if runs else []
    ms = {}
    for mk in margins:
        covs = np.array([r["ablations"]["dasp_margin_sensitivity"][mk]["coverage"]
                         for r in runs if mk in r["ablations"]["dasp_margin_sensitivity"]])
        ms[mk] = {"mean": float(covs.mean()), "std": float(covs.std(ddof=1)) if len(covs) > 1 else 0.0}
    agg["margin"] = ms

    # epistemic on/off ablation (the contribution-isolating numbers)
    ea = {}
    for key in ("epistemic_on", "epistemic_off"):
        vals = np.array([r["ablations"]["epistemic_ablation"][key]["aurc"]
                         for r in runs
                         if key in r.get("ablations", {}).get("epistemic_ablation", {})])
        if len(vals):
            ea[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    agg["epistemic_ablation"] = ea

    # std/PV/BALD estimator ablation (DASP AURC per epistemic estimator)
    est = {}
    for kind in ("std", "pv", "bald"):
        vals = np.array([r["ablations"]["estimator_ablation"][kind]["aurc"]
                         for r in runs
                         if kind in r.get("ablations", {}).get("estimator_ablation", {})])
        if len(vals):
            est[kind] = {"mean": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    agg["estimator_ablation"] = est

    # standard selective-prediction metrics (RPP, Refinement) per regime
    sm = {}
    sm_regimes = list(runs[0].get("selective_metrics", {}).keys()) if runs else []
    for regime in sm_regimes:
        rpp = np.array([r["selective_metrics"][regime]["rpp"] for r in runs
                        if regime in r.get("selective_metrics", {})])
        ref = np.array([r["selective_metrics"][regime]["refinement"] for r in runs
                        if regime in r.get("selective_metrics", {})])
        if len(rpp):
            sm[regime] = {
                "rpp_mean": float(rpp.mean()), "rpp_std": float(rpp.std(ddof=1)) if len(rpp) > 1 else 0.0,
                "ref_mean": float(ref.mean()), "ref_std": float(ref.std(ddof=1)) if len(ref) > 1 else 0.0,
            }
    agg["selective_metrics"] = sm

    # calibration (ECE before/after temperature scaling), averaged over seeds when present
    cals = [r.get("meta", {}).get("calibration") for r in runs]
    cals = [c for c in cals if c and c.get("ece_before") is not None
            and c.get("ece_after") is not None]
    if cals:
        eb = np.array([c["ece_before"] for c in cals])
        ef = np.array([c["ece_after"] for c in cals])
        agg["calibration"] = {
            "ece_before_mean": float(eb.mean()), "ece_after_mean": float(ef.mean()),
            "ece_before_std": float(eb.std(ddof=1)) if len(eb) > 1 else 0.0,
            "ece_after_std": float(ef.std(ddof=1)) if len(ef) > 1 else 0.0,
        }
    return agg


# --------------------------------------------------------------------------------
# Figure 1: headline Defense Risk-Coverage with bootstrap CI band
# --------------------------------------------------------------------------------
def fig_drc(summary: dict, fig_dir: str, dataset: str) -> list[str]:
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))  # single-column Elsevier width
    regimes = summary["regimes"]
    for regime in REGIME_ORDER:  # baselines first, DASP last -> drawn on top
        if regime not in regimes:
            continue  # e.g. SR-Unc absent when the run had no epistemic signal
        r = regimes[regime]
        cov = np.asarray(r["drc_common_coverage"])
        mean = np.asarray(r["drc_mean_risk"])
        lo = np.asarray(r["drc_ci_lo"])
        hi = np.asarray(r["drc_ci_hi"])
        st = REGIME_STYLE[regime]
        label = f"{regime} (AURC {r['aurc_mean']:.3f}$\\pm${r['aurc_std']:.3f})"
        ax.plot(cov, mean, color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["marker"], markersize=st["ms"], markevery=2, label=label, zorder=st["z"])
        # bootstrap CI band (only DASP + baselines get a light band; keep readable)
        ax.fill_between(cov, lo, hi, color=st["color"], alpha=0.15, linewidth=0, zorder=st["z"] - 1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Defense risk\n(missed actionable defense)")
    ax.set_title(f"Defense Risk-Coverage ({dataset.upper()})")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.01)
    return save_fig(fig, fig_dir, f"F1_DRC_{dataset}")


# --------------------------------------------------------------------------------
# Figure 2: epistemic on/off ablation (isolates the contribution)
# --------------------------------------------------------------------------------
def fig_epistemic(agg: dict, fig_dir: str, dataset: str) -> list[str] | None:
    ea = agg.get("epistemic_ablation", {})
    if "epistemic_on" not in ea or "epistemic_off" not in ea:
        return None  # e.g. TF-IDF baseline run has no epistemic signal
    # If on and off are identical, epistemic was unavailable (epi_norm=0 for all cases) and
    # the ablation is vacuous -> do not emit a misleading figure.
    if abs(ea["epistemic_on"]["mean"] - ea["epistemic_off"]["mean"]) < 1e-9:
        return None
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    labels = ["DASP\n(epistemic off)", "DASP\n(epistemic on)"]
    means = [ea["epistemic_off"]["mean"], ea["epistemic_on"]["mean"]]
    stds = [ea["epistemic_off"]["std"], ea["epistemic_on"]["std"]]
    colors = [CB["grey"], CB["vermillion"]]
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.6,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AURC (lower is better)")
    ax.set_title(f"Epistemic coupling ablation ({dataset.upper()})")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    # annotate the improvement
    if means[0] > 0:
        rel = 100.0 * (means[0] - means[1]) / means[0]
        ax.annotate(f"{rel:.0f}% lower", xy=(1, means[1]), xytext=(0.5, max(means) * 0.92),
                    ha="center", fontsize=9, color=CB["vermillion"])
    return save_fig(fig, fig_dir, f"F2_EPISTEMIC_{dataset}")


# --------------------------------------------------------------------------------
# Figure 3: subgroup-conditional reliability (mean +/- std across seeds)
# --------------------------------------------------------------------------------
def fig_subgroup(agg: dict, fig_dir: str, dataset: str) -> list[str] | None:
    sg = agg.get("subgroup", {})
    if not sg:
        return None
    set_pub_style()
    groups = list(sg.keys())
    cov_m = [sg[g]["cov_mean"] for g in groups]
    cov_s = [sg[g]["cov_std"] for g in groups]
    risk_m = [sg[g]["risk_mean"] for g in groups]
    risk_s = [sg[g]["risk_std"] for g in groups]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.bar(x - w / 2, cov_m, yerr=cov_s, width=w, label="coverage",
           color=CB["blue"], capsize=3, error_kw={"elinewidth": 1.0})
    ax.bar(x + w / 2, risk_m, yerr=risk_s, width=w, label="defense risk",
           color=CB["orange"], capsize=3, error_kw={"elinewidth": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("rate")
    ax.set_title(f"Subgroup-conditional reliability ({dataset.upper()})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    return save_fig(fig, fig_dir, f"F3_SUBGROUP_{dataset}")


# --------------------------------------------------------------------------------
# Figure 4: DASP margin sensitivity (mean +/- std across seeds)
# --------------------------------------------------------------------------------
def fig_margin(agg: dict, fig_dir: str, dataset: str) -> list[str] | None:
    ms = agg.get("margin", {})
    if not ms:
        return None
    set_pub_style()
    xs = sorted(float(k) for k in ms)
    means = [ms[_fmt_margin(x, ms)]["mean"] for x in xs]
    stds = [ms[_fmt_margin(x, ms)]["std"] for x in xs]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.errorbar(xs, means, yerr=stds, marker="o", markersize=4, linewidth=1.8,
                color=CB["vermillion"], capsize=3)
    ax.set_xlabel("DASP margin")
    ax.set_ylabel("Coverage")
    ax.set_title(f"DASP margin sensitivity ({dataset.upper()})")
    ax.grid(alpha=0.3, linewidth=0.5)
    return save_fig(fig, fig_dir, f"F4_MARGIN_{dataset}")


def _fmt_margin(x: float, ms: dict) -> str:
    """Find the original key for a margin value (keys were written as f'{margin:.2f}')."""
    target = f"{x:.2f}"
    if target in ms:
        return target
    # fallback: closest key
    return min(ms.keys(), key=lambda k: abs(float(k) - x))


# --------------------------------------------------------------------------------
# Figure 5: calibration (reliability) -- ECE before/after temperature scaling (Guo 2017)
# --------------------------------------------------------------------------------
def fig_calibration(agg: dict, fig_dir: str, dataset: str) -> list[str] | None:
    """
    Calibration evidence: Expected Calibration Error (Guo et al. 2017, 15 bins) before vs
    after temperature scaling, mean +/- std over seeds. A reliability diagram proper needs
    per-bin accuracy/confidence (not retained in the summary); this reports the ECE the
    diagram summarizes, which is the standard scalar calibration figure. Emitted only when
    calibration was recorded (InLegalBERT runs); skipped for the TF-IDF baseline.
    """
    cal = agg.get("calibration")
    if not cal:
        return None
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    labels = ["before\nscaling", "after\nscaling"]
    means = [cal["ece_before_mean"], cal["ece_after_mean"]]
    stds = [cal["ece_before_std"], cal["ece_after_std"]]
    colors = [CB["grey"], CB["green"]]
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.6,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"Calibration ({dataset.upper()})")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    if means[0] > 0:
        rel = 100.0 * (means[0] - means[1]) / means[0]
        ax.annotate(f"{rel:.0f}% lower", xy=(1, means[1]), xytext=(0.5, max(means) * 0.92),
                    ha="center", fontsize=9, color=CB["green"])
    return save_fig(fig, fig_dir, f"F5_CALIBRATION_{dataset}")


# --------------------------------------------------------------------------------
# Figure 6: epistemic estimator ablation -- DASP AURC for std vs PV vs BALD (Santosh 2024)
# --------------------------------------------------------------------------------
def fig_estimator(agg: dict, fig_dir: str, dataset: str) -> list[str] | None:
    """
    DASP AURC under each MC-dropout uncertainty estimator (predictive std, PV, BALD;
    Santosh Eq. 7-9), mean +/- std over seeds. Supports the choice of BALD as the default
    epistemic signal. Emitted only when the estimator ablation ran (InLegalBERT).
    """
    est = agg.get("estimator_ablation", {})
    kinds = [k for k in ("std", "pv", "bald") if k in est]
    if len(kinds) < 2:
        return None
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    means = [est[k]["mean"] for k in kinds]
    stds = [est[k]["std"] for k in kinds]
    # highlight BALD (the default) in the contribution color; others muted
    colors = [CB["vermillion"] if k == "bald" else CB["grey"] for k in kinds]
    x = np.arange(len(kinds))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.6,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in kinds])
    ax.set_ylabel("DASP AURC (lower is better)")
    ax.set_title(f"Epistemic estimator ({dataset.upper()})")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    return save_fig(fig, fig_dir, f"F6_ESTIMATOR_{dataset}")


# --------------------------------------------------------------------------------
# Tables (LaTeX, booktabs). Numbers are real; nothing is invented.
# --------------------------------------------------------------------------------
def _has_real_epistemic(agg: dict) -> bool:
    """True only if the epistemic ablation is non-vacuous (on differs from off)."""
    ea = agg.get("epistemic_ablation", {})
    if "epistemic_on" not in ea or "epistemic_off" not in ea:
        return False
    return abs(ea["epistemic_on"]["mean"] - ea["epistemic_off"]["mean"]) >= 1e-9


def table_headline(per_dataset: dict, out_path: str):
    """
    AURC mean +/- std per regime per dataset, actionable coverage, paired-test p-values.
    `per_dataset[ds] = {"summary": summary_dict_or_None, "agg": seed_agg_dict}`.
    """
    datasets = list(per_dataset.keys())
    lines = []
    lines.append("% Auto-generated by plots.py -- do not edit by hand.")
    lines.append("\\begin{table}[t]\\centering")
    lines.append("\\caption{Defense Risk-Coverage results. AURC is the area under the "
                 "Defense Risk-Coverage curve (lower is better), reported as "
                 "mean$\\pm$std over seeds. Actionable coverage is the fraction of cases "
                 "with a confident prediction or at least one legally-plausible "
                 "outcome-flipping counterfactual. $p$ is the paired Wilcoxon $p$-value of "
                 "DASP vs.\\ the baseline.}")
    lines.append("\\label{tab:headline}")
    ncol = "l" + "c" * len(datasets)
    lines.append("\\small\\begin{tabular}{" + ncol + "}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(d.upper() for d in datasets) + " \\\\")
    lines.append("\\midrule")

    # AURC per regime
    present_regimes = [rg for rg in REGIME_ORDER
                       if any(per_dataset[d]["summary"] and rg in per_dataset[d]["summary"]["regimes"]
                              for d in datasets)]
    for regime in present_regimes:
        cells = []
        for d in datasets:
            s = per_dataset[d]["summary"]
            if s and regime in s["regimes"]:
                r = s["regimes"][regime]
                txt = f"{r['aurc_mean']:.3f}$\\pm${r['aurc_std']:.3f}"
                if regime == "DASP":
                    txt = "\\textbf{" + txt + "}"
            else:
                txt = "--"
            cells.append(txt)
        name = "\\textbf{DASP (ours)}" if regime == "DASP" else regime
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    # actionable coverage
    cells = []
    for d in datasets:
        ac = per_dataset[d]["agg"].get("actionable_coverage")
        cells.append(f"{ac['mean']:.3f}$\\pm${ac['std']:.3f}" if ac else "--")
    lines.append("Actionable coverage & " + " & ".join(cells) + " \\\\")

    # paired-test p-values (DASP vs each baseline)
    for base in BASELINES:
        cells = []
        has_any = False
        for d in datasets:
            s = per_dataset[d]["summary"]
            key = f"DASP_vs_{base}"
            pt = s["paired_tests"].get(key) if s else None
            if pt and "p_value" in pt:
                cells.append(f"{pt['p_value']:.3f}")
                has_any = True
            elif pt and ("caveat" in pt or "note" in pt):
                cells.append("n/a$^\\dagger$")
                has_any = True
            else:
                cells.append("--")
        if has_any:
            lines.append(f"$p$ (DASP vs {base}) & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\\\[2pt]{\\footnotesize $^\\dagger$With $<6$ seeds the signed-rank test "
                 "cannot reach $p<0.05$; see the bootstrap CIs in Fig.~\\ref{fig:drc}.}")
    lines.append("\\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def table_epistemic(per_dataset: dict, out_path: str):
    """Epistemic on/off AURC per dataset (isolates the MC-dropout coupling)."""
    datasets = [d for d in per_dataset
                if _has_real_epistemic(per_dataset[d]["agg"])]
    lines = ["% Auto-generated by plots.py -- do not edit by hand.",
             "\\begin{table}[t]\\centering",
             "\\caption{Epistemic-coupling ablation: AURC of DASP with the MC-dropout "
             "epistemic signal enabled vs.\\ disabled ($w_{\\mathrm{epi}}{=}0$). "
             "mean$\\pm$std over seeds; lower is better.}",
             "\\label{tab:epistemic}",
             "\\small\\begin{tabular}{l" + "c" * len(datasets) + "}",
             "\\toprule",
             " & " + " & ".join(d.upper() for d in datasets) + " \\\\",
             "\\midrule"]
    if not datasets:
        lines.append("\\multicolumn{1}{l}{(no epistemic signal available)} \\\\")
    else:
        for key, name in (("epistemic_off", "DASP (epistemic off)"),
                          ("epistemic_on", "\\textbf{DASP (epistemic on)}")):
            cells = []
            for d in datasets:
                v = per_dataset[d]["agg"]["epistemic_ablation"][key]
                txt = f"{v['mean']:.3f}$\\pm${v['std']:.3f}"
                if "on" in key:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def table_selective(per_dataset: dict, out_path: str):
    """
    Standard selective-prediction metrics (RPP and Refinement; Santosh Eq. 4-5) per regime
    per dataset, mean$\\pm$std over seeds. Lets DASP be compared on the field's usual axis in
    addition to the defense-specific DRC. Lower is better for both (0 = ideal ranking).
    """
    datasets = [d for d in per_dataset if per_dataset[d]["agg"].get("selective_metrics")]
    lines = ["% Auto-generated by plots.py -- do not edit by hand.",
             "\\begin{table}[t]\\centering",
             "\\caption{Standard selective-prediction metrics: Reversed Pair Proportion (RPP) "
             "and Refinement (lower is better; $0$ = confidence ranking never contradicts "
             "correctness). mean$\\pm$std over seeds.}",
             "\\label{tab:selective}"]
    if not datasets:
        lines += ["\\end{table}"]
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return
    lines += ["\\small\\begin{tabular}{ll" + "c" * len(datasets) + "}",
              "\\toprule",
              "Metric & Regime & " + " & ".join(d.upper() for d in datasets) + " \\\\",
              "\\midrule"]
    for metric_key, metric_name in (("rpp", "RPP"), ("ref", "Refinement")):
        regimes = [rg for rg in REGIME_ORDER
                   if any(rg in per_dataset[d]["agg"].get("selective_metrics", {}) for d in datasets)]
        for ri, regime in enumerate(regimes):
            cells = []
            for d in datasets:
                sm = per_dataset[d]["agg"].get("selective_metrics", {}).get(regime)
                if sm:
                    txt = f"{sm[metric_key + '_mean']:.3f}$\\pm${sm[metric_key + '_std']:.3f}"
                    if regime == "DASP":
                        txt = "\\textbf{" + txt + "}"
                else:
                    txt = "--"
                cells.append(txt)
            label = metric_name if ri == 0 else ""
            rname = "\\textbf{DASP}" if regime == "DASP" else regime
            lines.append(f"{label} & {rname} & " + " & ".join(cells) + " \\\\")
        if metric_key == "rpp":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
def main():
    ap = argparse.ArgumentParser(description="Generate camera-ready figures/tables for DASP")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    found = discover(args.results)
    if not found:
        raise SystemExit(f"no result files found in {args.results} "
                         f"(expected summary_<ds>.json and/or results_<ds>_seed*.json)")

    manifest = {"figures": {}, "tables": {}}
    per_dataset = {}

    for dataset, paths in found.items():
        summary = load_json(paths["summary"]) if paths["summary"] else None
        agg = aggregate_seeds(paths["seeds"]) if paths["seeds"] else {}
        per_dataset[dataset] = {"summary": summary, "agg": agg}

        produced = []
        if summary is not None:
            produced += fig_drc(summary, fig_dir, dataset)
        else:
            print(f"[warn] {dataset}: no summary_{dataset}.json -> skipping headline DRC "
                  f"(run `experiment.py --aggregate --dataset {dataset}` first)")
        for fn in (fig_epistemic, fig_subgroup, fig_margin, fig_calibration, fig_estimator):
            res = fn(agg, fig_dir, dataset)
            if res:
                produced += res
        manifest["figures"][dataset] = {
            "produced": produced,
            "from_summary": paths["summary"],
            "from_seeds": paths["seeds"],
        }
        print(f"{dataset}: {len(produced)} figure files "
              f"(n_seeds={agg.get('n_seeds', 0)}, summary={'yes' if summary else 'no'})")

    # tables across datasets
    t1 = os.path.join(fig_dir, "table_headline.tex")
    t2 = os.path.join(fig_dir, "table_epistemic.tex")
    t3 = os.path.join(fig_dir, "table_selective.tex")
    table_headline(per_dataset, t1)
    table_epistemic(per_dataset, t2)
    table_selective(per_dataset, t3)
    manifest["tables"] = {"headline": t1, "epistemic": t2, "selective": t3}

    with open(os.path.join(fig_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"figures + tables written to {fig_dir} (provenance in manifest.json)")


if __name__ == "__main__":
    main()
