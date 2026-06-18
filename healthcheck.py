"""
healthcheck.py — mechanical verification that a real run is BEHAVING, not just running.

A pipeline can complete and still be silently wrong (uncalibrated probabilities, a
degenerate epistemic signal, a collapsed label distribution). This script reads the
result JSON and ASSERTS the health signals that separate a trustworthy run from a
plausible-but-wrong one. Intended to be run right after `experiment.py --aggregate`
(scripts/run_all.sh does this automatically).

Checks (each prints PASS/WARN/FAIL with the evidence):
  1. Test split is non-empty (positive-class balance is asserted in data.py at load).
  2. Calibration actually worked on InLegalBERT runs: temperature scaling ran AND the
     Expected Calibration Error decreased (ECE_after < ECE_before; Guo 2017). Falling back
     to a temperature != 1.0 check when ECE was not recorded.
  3. Epistemic signal is real on InLegalBERT runs => the epistemic-on/off ablation is
     non-vacuous (MC-dropout produced non-zero predictive variance), and the std/PV/BALD
     estimator ablation is present.
  4. DASP is not worse than ALL baselines (SR-Conf, SR-Unc, SR-CF) on mean AURC (sanity:
     the contribution should at least not hurt; a large regression signals a wiring problem).
  5. Subgroup coverage is reported for >1 group on ILDC (fairness axis is populated).
  6. Conformal coverage range is plausible (DASP coverage grid within [0,1]) — light sanity;
     full monotonicity is unit-tested in selective.py.

Exit code: 0 if no FAIL (WARNs allowed), non-zero if any FAIL. So CI/run_all can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def ok(self, msg):
        print(f"  {GREEN}PASS{RESET}  {msg}")

    def warn(self, msg):
        self.warns += 1
        print(f"  {YELLOW}WARN{RESET}  {msg}")

    def fail(self, msg):
        self.fails += 1
        print(f"  {RED}FAIL{RESET}  {msg}")


def _load(path):
    with open(path) as f:
        return json.load(f)


def check_per_seed(rep: Report, results_dir: str, dataset: str):
    seeds = sorted(f for f in os.listdir(results_dir)
                   if f.startswith(f"results_{dataset}_seed") and f.endswith(".json"))
    if not seeds:
        rep.fail(f"no per-seed result files for '{dataset}' in {results_dir}")
        return []
    runs = [_load(os.path.join(results_dir, f)) for f in seeds]
    rep.ok(f"found {len(runs)} per-seed result file(s) for '{dataset}'")

    for f, r in zip(seeds, runs):
        meta = r.get("meta", {})
        kind = meta.get("model_kind", "?")

        # (2) calibration: temperature moved AND ECE improved (Guo 2017). The strong signal
        #     is ECE_after < ECE_before; temperature != 1.0 alone only shows scaling ran.
        T = meta.get("temperature")
        cal = meta.get("calibration") or {}
        ece_b, ece_a = cal.get("ece_before"), cal.get("ece_after")
        if kind == "inlegalbert":
            if T is None:
                rep.fail(f"{f}: inlegalbert run but no temperature recorded "
                         f"(temperature scaling did not run -> probabilities uncalibrated)")
            elif ece_b is not None and ece_a is not None:
                if ece_a < ece_b - 1e-6:
                    rep.ok(f"{f}: calibration improved (ECE {ece_b:.3f} -> {ece_a:.3f}, T={T:.3f})")
                elif abs(ece_a - ece_b) <= 1e-6:
                    rep.warn(f"{f}: ECE unchanged by scaling ({ece_b:.3f}); the model may already "
                             f"be calibrated, or the dev set is too small to fit T.")
                else:
                    rep.fail(f"{f}: ECE WORSENED after scaling ({ece_b:.3f} -> {ece_a:.3f}); "
                             f"temperature fit is misbehaving (check the dev set).")
            elif abs(T - 1.0) < 1e-3:
                rep.warn(f"{f}: temperature == {T:.3f} (==1.0) and no ECE recorded. Calibration "
                         f"ran but had no effect; verify a non-trivial dev set was passed.")
            else:
                rep.ok(f"{f}: temperature = {T:.3f} (!=1.0 -> calibration active; ECE not recorded)")
        else:
            rep.ok(f"{f}: model_kind={kind} (temperature scaling N/A)")

        # (3) epistemic signal real?
        used = meta.get("epistemic_used", False)
        ea = r.get("ablations", {}).get("epistemic_ablation", {})
        on = ea.get("epistemic_on", {}).get("aurc")
        off = ea.get("epistemic_off", {}).get("aurc")
        if kind == "inlegalbert":
            if not used:
                rep.warn(f"{f}: epistemic_used=False on an inlegalbert run "
                         f"(use_mc_dropout disabled?). DASP runs in confidence-only mode.")
            elif on is not None and off is not None and abs(on - off) < 1e-9:
                rep.fail(f"{f}: epistemic on/off AURC identical ({on}); MC-dropout produced "
                         f"NO variance. Check enable_mc_dropout / dropout>0.")
            elif on is not None and off is not None:
                rep.ok(f"{f}: epistemic ablation non-vacuous (on={on:.4f}, off={off:.4f})")

            # (3b) estimator ablation present and reports BALD (the default epistemic signal)
            est = r.get("ablations", {}).get("estimator_ablation", {})
            if used and est:
                if "bald" in est:
                    kinds = ", ".join(f"{k}={est[k]['aurc']:.4f}" for k in ("std", "pv", "bald")
                                      if k in est)
                    rep.ok(f"{f}: estimator ablation present ({kinds})")
                else:
                    rep.warn(f"{f}: estimator ablation present but BALD (the default) missing")
        # TF-IDF: epistemic legitimately absent -> no check

    return runs


def check_summary(rep: Report, results_dir: str, dataset: str):
    path = os.path.join(results_dir, f"summary_{dataset}.json")
    if not os.path.exists(path):
        rep.warn(f"no summary_{dataset}.json (run `experiment.py --aggregate`); "
                 f"skipping cross-seed checks")
        return
    s = _load(path)
    regimes = s.get("regimes", {})

    # (4) DASP not catastrophically worse than the baselines present (incl. SR-Unc).
    baselines = [b for b in ("SR-Conf", "SR-Unc", "SR-CF") if b in regimes]
    if "DASP" in regimes and baselines:
        d = regimes["DASP"]["aurc_mean"]
        base_aurcs = {b: regimes[b]["aurc_mean"] for b in baselines}
        worst = max(base_aurcs.values())
        best = min(base_aurcs.values())
        summary_str = ", ".join(f"{b} {a:.4f}" for b, a in base_aurcs.items())
        if d <= best + 1e-9:
            rep.ok(f"DASP AURC {d:.4f} <= all baselines ({summary_str})")
        elif d <= worst:
            rep.warn(f"DASP AURC {d:.4f} beats some but not all baselines ({summary_str}); "
                     f"expected on some datasets")
        else:
            rep.fail(f"DASP AURC {d:.4f} WORSE than all baselines ({summary_str}); "
                     f"likely a wiring problem")
        # (6) coverage range sanity
        cov = regimes["DASP"].get("drc_common_coverage", [])
        if cov and (min(cov) < -1e-9 or max(cov) > 1 + 1e-9):
            rep.fail(f"coverage outside [0,1]: min={min(cov)}, max={max(cov)}")
        elif cov:
            rep.ok(f"coverage range valid [{min(cov):.2f}, {max(cov):.2f}]")
    else:
        rep.fail("summary missing DASP or all baseline regimes")

    # paired-test presence (informational)
    pt = s.get("paired_tests", {})
    for k, v in pt.items():
        if "p_value" in v:
            print(f"  ---   {k}: p={v['p_value']:.4f} (n={v.get('n')})")
        elif "caveat" in v:
            print(f"  ---   {k}: {v['caveat']}")


def check_label_and_subgroup(rep: Report, runs, dataset: str):
    if not runs:
        return
    r0 = runs[0]
    # (5) subgroup populated (ILDC should have >1 era group; ECtHR is single by design)
    sg = r0.get("subgroup", {})
    if dataset == "ildc":
        if len(sg) > 1:
            rep.ok(f"subgroup axis populated with {len(sg)} groups: {sorted(sg)}")
        else:
            rep.warn(f"ILDC subgroup has {len(sg)} group(s); expected era buckets "
                     f"(pre1990/1990_2010/post2010). Check id formatting.")
    else:
        rep.ok(f"{dataset}: subgroup is single-bucket by design (documented)")
    # (1) test split non-empty. The explicit positive-class balance is printed by data.py at
    #     load time and asserted there; here we verify the split is non-empty (a wiring check).
    n = r0.get("meta", {}).get("n_test", 0)
    if n <= 0:
        rep.fail(f"n_test={n}; the test split is empty")
    else:
        rep.ok(f"n_test={n} (non-empty). Positive-class balance is asserted in the [data] log lines.")


def main():
    ap = argparse.ArgumentParser(description="DASP run health check")
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="ildc")
    args = ap.parse_args()

    print(f"=== health check: dataset={args.dataset}, results={args.results} ===")
    rep = Report()
    runs = check_per_seed(rep, args.results, args.dataset)
    check_label_and_subgroup(rep, runs, args.dataset)
    check_summary(rep, args.results, args.dataset)

    print(f"\n=== summary: {rep.fails} FAIL, {rep.warns} WARN ===")
    if rep.fails:
        print(f"{RED}Run is NOT trustworthy until FAILs are resolved.{RESET}")
        sys.exit(1)
    if rep.warns:
        print(f"{YELLOW}Run is usable; review WARNs.{RESET}")
    else:
        print(f"{GREEN}All health signals nominal.{RESET}")
    sys.exit(0)


if __name__ == "__main__":
    main()
