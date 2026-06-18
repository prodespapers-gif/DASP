# CASP — Counsel-Aware Selective Prediction for Appellant-Side Legal Outcome Prediction

Reference implementation for the paper
**“Beyond Prosecution-Centric AI: A Trustworthy Expert System for Appellant-Side Legal Outcome Prediction with Selective Abstention and Counterfactual Explanation.”**

This repository is shared for **double-blind peer review**. It contains no author names, institutional identifiers, or licence keys. Please do not de-anonymize the authors.

> **Naming note.** The paper calls the method **CASP (Counsel-Aware Selective Prediction)**. The source code predates the final naming and still uses the identifier **DASP** (and the word *defense* in the sense of *defence counsel / the represented party’s advocate*) for the same method, the same `SelectivePredictor` regime string, and the same result fields. **`DASP` in the code == `CASP` in the paper.** “Defense” throughout the code means **the defendant/appellant’s own counsel**, an advocate for the represented party — it has no relation to military or defense systems.

---

## 1. What this is

A **selective-prediction expert system for Legal Judgment Prediction (LJP)**, built for the side of the *represented party’s counsel*. Instead of always emitting an outcome, the system **abstains** when deciding could harm the client, and — when it predicts against the client — it can surface the **smallest legally admissible counterfactual edit** that would flip the outcome in the client’s favour, for a human to review.

The core thesis: the dangerous case is not “the model is unsure,” but **“the model is confident the client loses, yet a legally admissible argument would reverse it.”** A confidence-only gate is structurally blind to exactly that case. CASP is built to catch it by coupling three per-case signals:

- **p** — a temperature-calibrated outcome probability;
- **epi** — Monte-Carlo-dropout **epistemic uncertainty** (predictive std / PV / BALD; BALD by default);
- **r = 1 − d\*** — **counterfactual reachability**, where d\* is the smallest *legally plausible* edit distance that flips the prediction to favourable.

CASP abstains when a confident unfavourable prediction is **contradicted** by a reachable favourable outcome relative to the model’s residual certainty, and hands counsel the supporting counterfactual rather than silently conceding a winnable case.

---

## 2. Repository layout

| File | Role |
|---|---|
| `data.py` | Loads and normalizes the two corpora into uniform case records (text, label, subgroup, optional legal-NER spans). Reports split statistics. |
| `model.py` | InLegalBERT hierarchical encoder; temperature calibration; **MC-dropout** epistemic estimators (std / PV / BALD); model-tagged feature cache. Supplies the learned plausibility scorer used by the filter. |
| `plausibility.py` | **Contribution 1.** Legal edit-type taxonomy (mutable vs immutable) + the two-tier `LegalPlausibilityFilter`; immutable-span detection from legal-NER + date/quantity regex; the legal-type → Polyjuice control-code bridge. |
| `counterfactual.py` | Generates candidate edits (Polyjuice), filters them through the plausibility filter, and computes the per-case **reachability** `r = 1 − d*`. Disk cache with optional **version-binding** so stale counterfactuals cannot leak across model/generator revisions. |
| `selective.py` | **Contribution 2.** `SelectivePredictor` with the four regimes (`SR-Conf`, `SR-Unc`, `SR-CF`, `DASP`=CASP), the coupling rule, and `ConformalConfidence` (split / per-subgroup Mondrian conformal). |
| `metrics.py` | Calibration (ECE), standard selective-prediction metrics (AURC, RPP, Refinement), and the novel **Counsel-Gain Risk–Coverage** metric / AU-CG-RC. |
| `experiment.py` | Orchestrator. Wires data → model → counterfactuals → selective regimes → metrics; supplies the real Polyjuice wrapper; multi-seed runs with `--aggregate`. |
| `healthcheck.py` | Mechanical post-run verification that a run is *behaving*, not merely *completing*. Exits non-zero on any FAIL so CI/automation can gate. |
| `plots.py` | Figure generation from result files. |
| `__init__.py` | Package marker (empty). |
| `DATA_CARD.md` | Dataset documentation, provenance, licences, and ethics notes. |

Every core module has a **CPU-only `__main__` self-test** that runs without downloading a model or a dataset (see §5).

---

## 3. Method at a glance

### Three signals → one coupled decision
For a case the model predicts unfavourable (`p < 0.5`, so residual `ρ = min(p, 1−p)`), CASP forms a counsel signal and abstains when it materially exceeds the model’s residual certainty:

```
counsel_signal = r + w_epi * epi_norm
abstain   ⇔   (p < 0.5)  AND  ( counsel_signal − ρ ) > margin
```

- `epi_norm ∈ [0,1]` is epistemic uncertainty placed on a common axis via a per-subgroup calibration scale.
- `w_epi` is a **fixed** config weight (not swept). `margin` is the **only** swept knob, which keeps CASP a single-parameter method whose risk–coverage curve is directly comparable to the baselines.
- **Backward-compatible:** with `epi = None`, the rule reduces *exactly* to the confidence-only coupling.

### The four selective regimes (each swept on exactly one knob)
| Regime (code id) | Abstains iff | Knob |
|---|---|---|
| `SR-Conf` | `conf < τ` (split-conformal, finite-sample coverage) | `tau_conf` |
| `SR-Unc` | `epi > τ` (MC-dropout uncertainty gate) | `tau_unc` |
| `SR-CF` | `r > τ` (reachability only) | `tau_reach` |
| **`DASP` (= CASP)** | the coupled rule above | `margin` |

Having **both** a confidence baseline and an MC-uncertainty baseline means the comparison is not against a single weak gate.

### Legal plausibility (why reachability is meaningful, not lexical)
A counterfactual that flips the outcome is useless unless the **edit it represents** is something counsel could actually argue. The filter has two tiers:

- **Tier A (hard rules):** reject any edit that overlaps an **immutable span** (statutes, precedents, dates, case numbers, court/judge/authority, plus regex-detected dates and quantities) **or** is classified as an **immutable edit type** (prior conviction, co-defendant confession, undisputed physical fact, statutory text).
- **Tier B (learned, optional):** score plausibility with an InLegalBERT-based classifier (injected, so this module stays framework-light and unit-testable).

Mutable edit types (witness credibility, evidence admissibility, mitigating circumstance, procedural error, alternative interpretation) are admissible. The **asymmetry is deliberate**: a mutable edit wrongly blocked costs one candidate; an immutable edit wrongly admitted would let a legally vacuous edit pose as a recommendation. Reachability is defined over **plausible edits only**.

### Conformal coverage, per subgroup
`ConformalConfidence` implements split conformal (Angelopoulos & Bates, 2021) with the finite-sample quantile level `⌈(n+1)(1−α)⌉ / n`. With `per_subgroup=True` it fits a separate threshold within each subgroup (Mondrian / group-conditional), so the coverage guarantee holds **within** subgroups, and the same machinery scales the epistemic signal per subgroup.

---

## 4. Installation

Python 3.10+ recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The self-tests need only `numpy`. A full experiment additionally needs `torch`, `transformers` (for InLegalBERT), the Polyjuice generator, and the dataset libraries listed in `DATA_CARD.md`. The pipeline is **CPU-capable** (smaller batches / cached features); a GPU is recommended for full runs.

---

## 5. Quick start: run the self-tests (no model, no data)

Each core module verifies its own contract on CPU in seconds:

```bash
python plausibility.py     # taxonomy, immutable spans, UNKNOWN policy, control-code bridge
python selective.py        # coupling rule, SR-Unc baseline, backward-compat, monotonicity,
                           # single-knob coverage, per-subgroup conformal
python counterfactual.py   # reachability math, no-flip, immutable-only, cache versioning
python healthcheck.py      # (run after a real experiment; see below)
```

These assert the **paper’s formal properties** directly: abstention is monotone in `epi` and in `r`; coverage is monotone in `margin` (well-posed single-knob sweep); and with `epi=None` CASP coincides exactly with the confidence rule.

---

## 6. Reproducing the experiments

```bash
# 1) one corpus, one seed (writes a results JSON)
python experiment.py --dataset ildc  --seed 0 --out results/ildc_seed0.json

# 2) all seeds, then aggregate to the tables/figures inputs
python experiment.py --dataset ildc  --aggregate --out results/ildc/
python experiment.py --dataset ecthr --aggregate --out results/ecthr/

# 3) gate the run (exits non-zero on any FAIL)
python healthcheck.py results/ildc/aggregate.json

# 4) figures
python plots.py results/ildc/aggregate.json --outdir figs/
```

(Exact flags are documented in `experiment.py --help`; dataset names follow `DATA_CARD.md`.)

`healthcheck.py` asserts, among other things: a non-empty test split; that temperature scaling **improved** ECE on InLegalBERT runs; that the MC-dropout signal is **non-degenerate** (so the epistemic ablation is non-vacuous); that CASP is **not worse than all baselines** on AURC; that subgroup coverage is populated; and that conformal coverage stays in `[0,1]`.

---

## 7. Reproducibility notes

- **Seeds.** Results in the paper are averaged over multiple seeds; `--aggregate` collects them and reports mean ± SD. Set the seed list in `experiment.py` (or pass `--seed` per run).
- **Determinism.** Given the per-case signals, every selective rule is **pure** — all stochasticity is upstream in `model.py`’s MC sampling — so risk–coverage curves are reproducible for a fixed seed.
- **Caching.** Feature encodings are **model-tagged** and counterfactuals are **version-tagged**; partial reruns are inexpensive and stale artifacts cannot leak across revisions. Delete the cache directories to force a clean recompute.
- **Ablations** are first-class: estimator choice (`std`/`PV`/`BALD`), epistemic coupling on/off, and `rules_only` vs `rules+learned` plausibility are all switchable without code edits.

---

## 8. Datasets

See `DATA_CARD.md` for full provenance, licences, and access. In brief:

- **ILDC / CJPE** (primary) — Indian Legal Documents Corpus, accessed via IL-TUR; **gated** access under its own terms (CC-BY-NC-SA-4.0). The favourable-outcome label is taken from the represented party’s perspective.
- **ECtHR via LexGLUE** (secondary) — cross-jurisdiction; a positive label (at least one violation found) corresponds to the applicant’s win. Openly available under its stated licence.

Raw corpora are **not** redistributed here. `data.py` loads them from their official sources into the uniform record format the pipeline expects.

---

## 9. Ethics and intended use

This is a **decision-support** tool for legal professionals, not a decision-maker. CASP is explicitly designed to **abstain and defer to a human** on the cases where prediction is most consequential, and to surface a reviewable argument rather than a verdict. Predicted outcomes and generated counterfactuals are hypotheses for counsel to evaluate, not legal advice. Court records carry sensitive information; users must comply with the licences and the privacy terms documented in `DATA_CARD.md`.

---

## 10. Licence

Code released for review under a permissive open-source licence (finalized on de-anonymization). Dataset licences are governed by their respective providers; see `DATA_CARD.md`.
