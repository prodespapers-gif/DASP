"""
counterfactual.py  —  counterfactual generation, filtered and cached.

Generates candidate edits with Polyjuice (Wu et al. 2021), runs them through the
LegalPlausibilityFilter (plausibility.py), measures how small the *plausible* flipping
edit is, and produces the per-case reachability signal d* / reach consumed by selective.py.

reach = 1 - d*, where d* is the minimal normalized edit distance of a LEGALLY PLAUSIBLE
edit that flips the prediction to favorable. This operationalizes Wachter, Mittelstadt &
Russell (2018): a counterfactual is "the smallest change to the world that can be made to
obtain a desirable outcome" -- restricted here to legally-admissible edits. Minimality is
enforced by taking the MINIMUM distance over plausible flips, which is Polyjuice's
closeness filter in effect (Wu et al. 2021, Sec. 4).

Edit-type labeling: each generated edit is classified (plausibility.classify_edit_type)
and the type is attached to the candidate, so the legal taxonomy is populated and the
filter can reject immutable EDIT TYPES, not only edits that physically touch immutable
spans. To STEER generation toward edits the filter will accept, experiment.py's Polyjuice
wrapper can request the control codes from polyjuice_codes_for() for each mutable type
(the linguistic-code <-> legal-type bridge defined in plausibility.py).

CPU strategy (critical): Polyjuice is GPT-2-scale and slow on CPU. We therefore
GENERATE ONCE, OFFLINE, and cache every (case_id -> [filtered candidates, distances]) to
disk (data/cf_cache/). Re-runs of the experiment read the cache and cost seconds. The
generation pass is checkpointed so it can resume. An optional cache_version binds the
cache to the model/generator version so stale counterfactuals can never silently leak
across experiments (cf. the model-tagged feature cache in model.py).

Reference: Wachter, Mittelstadt & Russell. Counterfactual Explanations Without Opening the
Black Box. Harvard JOLT 2018 (reachability = smallest change to a desirable outcome).
Wu, Ribeiro, Heer & Weld. Polyjuice. ACL-IJCNLP 2021 (generation + closeness filter).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

from plausibility import (LegalPlausibilityFilter, EditCandidate, Span,
                          detect_immutable_spans, classify_edit_type,
                          polyjuice_codes_for, EditType)

# Re-exported so experiment.py can steer Polyjuice without importing plausibility directly.
__all__ = ["CFResult", "normalized_edit_distance", "CounterfactualEngine",
           "polyjuice_codes_for", "EditType"]


@dataclass
class CFResult:
    case_id: str
    flipped: bool                 # did any plausible edit flip the prediction to favorable?
    min_distance: float           # normalized edit distance of closest plausible flip (1.0 if none)
    n_plausible: int              # number of plausible (filter-passing) candidates
    best_edit: str | None         # the closest plausible edited text (for qualitative tables)
    best_edit_type: str | None = None  # the edit type of the closest plausible flip

    @property
    def reachability(self) -> float:
        """1 - d* in [0,1]. 0 when no plausible flip exists (min_distance defaults to 1.0)."""
        return 1.0 - self.min_distance


def normalized_edit_distance(a: str, b: str) -> float:
    """
    Token-level Levenshtein / max(len) in [0,1]. Cheap, CPU-friendly, and symmetric. This is
    the distance d in d* = min over plausible flips; smaller = a more minimal (more
    actionable) edit, matching Wachter's "smallest change" and Polyjuice's closeness notion.
    """
    ta, tb = a.split(), b.split()
    n, m = len(ta), len(tb)
    if n == 0 and m == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ta[i - 1] != tb[j - 1]))
            prev = cur
    return dp[m] / max(n, m)


def _coerce_spans(ner_spans) -> list[Span] | None:
    """
    Normalize ner_spans into list[Span]. Accepts None, a list of Span, or a list of
    (start, end, label) tuples. Malformed entries are skipped rather than crashing.
    """
    if not ner_spans:
        return None
    out: list[Span] = []
    for s in ner_spans:
        if isinstance(s, Span):
            out.append(s)
        elif isinstance(s, (tuple, list)) and len(s) == 3:
            out.append(Span(int(s[0]), int(s[1]), str(s[2])))
        # else: skip silently-malformed span entry
    return out or None


class CounterfactualEngine:
    """
    Wraps a generator (Polyjuice) + a predictor + the plausibility filter.
    `generate_fn(text) -> list[(edited_text, edit_start, edit_end)]` is injected so this
    file is import-light and testable without loading Polyjuice; experiment.py supplies the
    real Polyjuice-backed callable. `predict_fn(text) -> p_favorable` is the trained model.

    Each generated edit is classified into the legal taxonomy and the type is attached to
    the candidate before filtering, so immutable edit TYPES are rejected (not only edits
    that overlap immutable spans). With a classifier in place, the stricter UNKNOWN policy
    (LegalPlausibilityFilter(admit_unknown=False)) becomes usable.

    Optional behavior (defaults preserve the original semantics):
      - cache_version : a string binding the cache to a model/generator version. When set,
                        a cache written under a different version is discarded on load, so
                        stale counterfactuals cannot leak across experiments.
      - skip_if_favorable : when True, cases the model already predicts favorable
                        (p >= flip_threshold on the ORIGINAL text) get reach = 0 without a
                        CF search -- reachability only guards predicted-LOSS cases, and DASP
                        ignores reachability when p >= 0.5 anyway, so this only saves work.
    """

    def __init__(self, generate_fn, predict_fn, plausibility: LegalPlausibilityFilter,
                 cache_path: str, flip_threshold: float = 0.5,
                 cache_version: str | None = None, skip_if_favorable: bool = False):
        self.generate_fn = generate_fn
        self.predict_fn = predict_fn
        self.filter = plausibility
        self.cache_path = cache_path
        self.flip_threshold = flip_threshold
        self.cache_version = cache_version
        self.skip_if_favorable = skip_if_favorable
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    def _assess_case(self, case_id: str, text: str, ner_spans: list) -> CFResult:
        # Optional fast path: reachability is only meaningful for predicted-loss cases.
        if self.skip_if_favorable and self.predict_fn(text) >= self.flip_threshold:
            return CFResult(case_id, False, 1.0, 0, None, None)

        spans = detect_immutable_spans(text, _coerce_spans(ner_spans))
        raw = self.generate_fn(text)
        cands = []
        for item in raw:
            if not (isinstance(item, (tuple, list)) and len(item) == 3):
                continue  # skip a malformed (edited, start, end) entry
            ed, st, en = item
            etype = classify_edit_type(ed)  # populate the legal taxonomy
            cands.append(EditCandidate(text, ed, int(st), int(en), edit_type=etype))

        kept = self.filter.filter(cands, spans)        # plausible only
        best_d, best_edit, best_type, flipped = float("inf"), None, None, False
        for cand, res in kept:
            p = self.predict_fn(cand.edited)
            if p >= self.flip_threshold:               # flipped to favorable
                d = normalized_edit_distance(cand.original, cand.edited)
                if d < best_d:
                    best_d, best_edit, flipped = d, cand.edited, True
                    best_type = res.edit_type.value if res.edit_type else None
        # min_distance is 1.0 when nothing flipped (reach = 0); otherwise the smallest flip.
        min_distance = best_d if flipped else 1.0
        return CFResult(case_id, flipped, min_distance, len(kept), best_edit, best_type)

    def run(self, examples, resume: bool = True) -> dict[str, CFResult]:
        cache = self._load_cache() if resume else {}
        records = cache.get("records", {}) if cache else {}
        n_since_save = 0
        for ex in examples:
            if ex.id in records:
                continue
            ner = getattr(ex, "ner_spans", None)
            records[ex.id] = asdict(self._assess_case(ex.id, ex.text, ner))
            n_since_save += 1
            if n_since_save >= 1:           # checkpoint each case (resumable)
                self._save_cache(records)
                n_since_save = 0
        self._save_cache(records)
        return {k: CFResult(**v) for k, v in records.items()}

    def _load_cache(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}
        with open(self.cache_path) as f:
            blob = json.load(f)
        # Versioned cache: discard if the stored version does not match (stale CFs can't leak).
        if isinstance(blob, dict) and "records" in blob:
            if self.cache_version is not None and blob.get("version") != self.cache_version:
                return {}
            return blob
        # Back-compat: an older cache was a bare {case_id: record} map with no wrapper.
        if self.cache_version is None:
            return {"records": blob}
        return {}  # versioning requested but cache predates it -> regenerate

    def _save_cache(self, records: dict):
        blob = {"version": self.cache_version, "records": records}
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)


if __name__ == "__main__":
    # Self-test with toy generator/predictor (no Polyjuice, no model) -- verifies the
    # reachability math, edit-type classification, and the plausibility gating on CPU.
    def fake_generate(text):
        # one immutable-touching edit (date) and one mutable witness edit
        out = []
        if "2001" in text:
            i = text.index("2001"); out.append((text.replace("2001", "2010"), i, i + 4))
        if "witness" in text:
            out.append(("acquitted because the witness testimony was uncorroborated " + text, 0, 0))
        return out

    def fake_predict(text):       # 'acquitted' present -> favorable
        return 0.9 if "acquitted" in text else 0.2

    filt = LegalPlausibilityFilter(rules_only=True)
    eng = CounterfactualEngine(fake_generate, fake_predict, filt,
                               cache_path="/tmp/cf_cache/test.json")

    class E:  # minimal example stand-in
        id = "c1"; text = "The appellant was convicted on 5th May 2001 per witness PW-1."; ner_spans = []
    res = eng.run([E()], resume=False)["c1"]
    assert res.flipped and res.reachability > 0, "favorable edit should be reachable"
    assert 0.0 < res.min_distance < 1.0
    # the flipping edit is classified as witness_credibility (taxonomy populated)
    assert res.best_edit_type == EditType.WITNESS_CREDIBILITY.value

    # ---- no-flip case: nothing flips -> reach = 0, flipped False ----
    def gen_noflip(text):
        return [("a slightly reworded version of " + text, 0, 0)]
    eng2 = CounterfactualEngine(gen_noflip, fake_predict, filt, cache_path="/tmp/cf_cache/nf.json")

    class E2:
        id = "c2"; text = "The appeal was dismissed."; ner_spans = []
    r2 = eng2.run([E2()], resume=False)["c2"]
    assert r2.flipped is False and r2.reachability == 0.0 and r2.min_distance == 1.0

    # ---- immutable-only edits: all filtered out -> no plausible candidates ----
    def gen_immutable(text):
        i = text.index("2001"); return [(text.replace("2001", "1999"), i, i + 4)]
    eng3 = CounterfactualEngine(gen_immutable, fake_predict, filt, cache_path="/tmp/cf_cache/im.json")

    class E3:
        id = "c3"; text = "Convicted on 5th May 2001."; ner_spans = []
    r3 = eng3.run([E3()], resume=False)["c3"]
    assert r3.n_plausible == 0 and r3.flipped is False

    # ---- malformed generator output is skipped, not fatal ----
    def gen_malformed(text):
        return [("acquitted " + text, 0, 0), ("bad tuple", 0), "not a tuple", (1, 2, 3, 4)]
    eng4 = CounterfactualEngine(gen_malformed, fake_predict, filt, cache_path="/tmp/cf_cache/mf.json")

    class E4:
        id = "c4"; text = "Convicted per witness."; ner_spans = []
    r4 = eng4.run([E4()], resume=False)["c4"]
    assert r4.flipped is True  # the one valid edit still works

    # ---- distance symmetry ----
    assert abs(normalized_edit_distance("a b c", "a b c d")
               - normalized_edit_distance("a b c d", "a b c")) < 1e-12

    # ---- cache versioning: a version mismatch discards the stale cache ----
    import os as _os
    vpath = "/tmp/cf_cache/ver.json"
    if _os.path.exists(vpath):
        _os.remove(vpath)
    engv1 = CounterfactualEngine(fake_generate, fake_predict, filt, cache_path=vpath,
                                 cache_version="model_v1")
    engv1.run([E()], resume=True)
    engv2 = CounterfactualEngine(fake_generate, fake_predict, filt, cache_path=vpath,
                                 cache_version="model_v2")  # different version
    loaded = engv2._load_cache()
    assert loaded == {}, "stale cache (different version) must be discarded"

    # ---- ner_spans coercion: tuples and Span objects both accepted ----
    assert _coerce_spans([(0, 5, "STAT")])[0].label == "STAT"
    assert _coerce_spans([Span(0, 5, "PREC")])[0].label == "PREC"
    assert _coerce_spans([("bad",), (1, 2, 3)])[0].start == 1  # malformed skipped, valid kept
    assert _coerce_spans(None) is None

    # ---- polyjuice steering bridge is reachable from here ----
    assert polyjuice_codes_for(EditType.WITNESS_CREDIBILITY)  # non-empty for mutable type

    print(f"counterfactual.py self-test passed (reach={res.reachability:.2f}, "
          f"edit_type={res.best_edit_type}, no-flip/immutable/malformed/versioning verified)")
