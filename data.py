"""
data.py
================================================================================
Dataset acquisition and deterministic preparation for the DASP system.

Loads the two corpora used in the paper and returns uniform `Example` records:

  * ILDC (primary, defense-native) via the IL-TUR benchmark, task `cjpe`.
    Label is the appellant/defense outcome: 1 = ACCEPTED (appellant wins),
    0 = REJECTED. A 56-document subset carries expert explanation annotations.
  * ECtHR (secondary, generalisation) via LexGLUE.

--------------------------------------------------------------------------------
VERIFIED IL-TUR cjpe SCHEMA (from the IL-TUR task documentation; the loader is
written against this exact shape):

    { 'id'   : string,                       # IndianKanoon case id
      'text' : string,                       # case contents (a few loaders store a
                                             #   list of sentences; we coerce either)
      'label': ClassLabel,                   # final ACCEPT/REJECT decision (0/1)
      'expert_1': { 'label': ClassLabel,     # that expert's decision
                    'rank_1': List(string),  # sentences (NOTE: 'rank_1' WITH underscore)
                    'rank_2': List(string), ... },
      'expert_2': {...}, ... }

Two schema points are easy to get wrong and are handled explicitly: the expert
salient-sentence keys are 'rank_1'..'rank_5' (underscore), and `text` may arrive
as a list of sentences. Getting either wrong silently yields empty explanations or
empty text, so both are coerced and the self-test covers them.

--------------------------------------------------------------------------------
LABEL SEMANTICS -- consistency across corpora (this is a correctness point, not a
detail). Throughout the system, label = 1 denotes the outcome FAVORABLE TO THE
PARTY THE SYSTEM SERVES (the appellant/applicant -- the defense-analog):

  * ILDC:  1 = appeal ACCEPTED         (the appellant prevails).
  * ECtHR: 1 = at least one VIOLATION FOUND (the applicant prevails against the
           respondent State). NOTE: "violation found" is the *applicant's* win,
           hence the positive class -- consistent with ILDC. Labelling "no
           violation" as positive would make the respondent State (the
           prosecution-analog) the winner, contradicting this paper's thesis.

--------------------------------------------------------------------------------
ROBUSTNESS / REPRODUCIBILITY:
  * Labels are coerced from either integer or string ClassLabel encodings and
    validated to be in {0, 1}; a MISSING label fails loudly rather than silently
    becoming 0, and malformed labels raise rather than collapsing to 0.
  * The ILDC loader tries the documented `revision="script"` first, then falls
    back; a gating error produces an actionable message (accept terms + HF_TOKEN).
  * If the configured split names do not match the dataset (so the splits come back
    empty), the loader raises a LOUD error listing the dataset's ACTUAL split names,
    instead of proceeding with no data. Split names are named constants for easy fix.
  * Processed splits are cached to disk (JSONL) keyed by dataset+config+version, so
    repeated runs skip re-download and re-flattening.
  * Optional `max_samples` (seeded) supports fast smoke runs; it is OFF by default
    and, when used, is recorded so results are never silently sub-sampled.
  * `content_hash` fingerprints id + label + a text digest, so two different texts
    cannot collide.

The expert explanation annotations preserve their per-expert, per-rank structure
(the paper evaluates counterfactual fidelity against ranked salient sentences); a
flat list is derivable when a consumer wants one.

MEMORY: with 384 GB RAM, materialising ~34k `Example` records (long documents) is
comfortable, so splits are returned as lists.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------
# IL-TUR split-name configuration. These are the assumed split names for cjpe; if a
# future IL-TUR revision renames them, correct them HERE (one place) and the loud
# empty-split check below will have told you the actual names to use.
# --------------------------------------------------------------------------------
ILDC_TRAIN_SPLITS = ["multi_train", "single_train"]
ILDC_DEV_SPLITS = ["multi_dev", "single_dev"]
ILDC_TEST_SPLITS = ["test"]
ILDC_EXPERT_SPLITS = ["expert"]

# IL-TUR L-NER label set (12 fine-grained legal entity types), in dataset label-index
# order. Used to translate integer span labels into the names plausibility.py matches.
L_NER_LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
                "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]


# --------------------------------------------------------------------------------
# Record type (fields consumed downstream: id, text, label, subgroup, ner_spans,
# gold_explanation). Field order/names are part of the package contract.
# --------------------------------------------------------------------------------
@dataclass
class Example:
    id: str
    text: str
    label: int                     # 1 = favorable to appellant/applicant (defense-analog)
    subgroup: str                  # disaggregation axis (e.g. era bucket) for fairness
    ner_spans: list = field(default_factory=list)        # (start, end, label) immutable spans
    gold_explanation: dict = field(default_factory=dict)  # {expert_i: {rank_j: [sentences]}}

    def flat_explanation(self) -> list[str]:
        """All gold salient sentences as a flat list (rank/expert structure discarded)."""
        return [s for exp in self.gold_explanation.values()
                for sents in exp.values() for s in sents]


# --------------------------------------------------------------------------------
# Label / text / NER coercion
# --------------------------------------------------------------------------------
_ILDC_LABEL_MAP = {
    "REJECTED": 0, "ACCEPTED": 1,
    "rejected": 0, "accepted": 1,
    "0": 0, "1": 1,
}


def coerce_label(raw) -> int:
    """
    Map a HF ClassLabel (int OR string) to {0,1}. Fails loudly on anything else,
    so a schema surprise can never silently collapse all labels to 0.
    """
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        if raw in (0, 1):
            return raw
        raise ValueError(f"unexpected integer label {raw} (expected 0 or 1)")
    if isinstance(raw, str):
        key = raw.strip()
        if key in _ILDC_LABEL_MAP:
            return _ILDC_LABEL_MAP[key]
        raise ValueError(f"unknown string label {raw!r}")
    raise TypeError(f"unsupported label type {type(raw).__name__}: {raw!r}")


def _require_label(row) -> int:
    """Read and coerce the row's label, failing loudly if the field is absent."""
    if "label" not in row:
        raise KeyError("row has no 'label' field -- schema mismatch; refusing to default to 0")
    return coerce_label(row["label"])


def coerce_text(text) -> str:
    """Coerce text that may be a string or a list of sentences into a single string."""
    if isinstance(text, list):
        return " ".join(str(t) for t in text)
    return text or ""


def ner_label_name(label) -> str:
    """Translate an integer L-NER class label into its name; pass names through unchanged."""
    if isinstance(label, bool):
        return str(label)
    if isinstance(label, int) and 0 <= label < len(L_NER_LABELS):
        return L_NER_LABELS[label]
    return str(label)


def _coerce_ner_spans(raw_spans) -> list:
    """
    Normalize raw L-NER spans into (start, end, label_name) tuples. Accepts dicts
    ({'start','end','label'}) or tuples; integer labels are mapped to names so the
    downstream plausibility filter (which matches names) works. Empty/None -> [].
    """
    if not raw_spans:
        return []
    out = []
    for s in raw_spans:
        if isinstance(s, dict):
            start, end, label = s.get("start"), s.get("end"), s.get("label")
        elif isinstance(s, (tuple, list)) and len(s) == 3:
            start, end, label = s
        else:
            continue
        if start is None or end is None:
            continue
        out.append((int(start), int(end), ner_label_name(label)))
    return out


# --------------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------------
def load_ildc(hf_token: str | None, cache_dir: str,
              max_samples: int | None = None, seed: int = 13,
              processed_cache: str | None = None) -> dict[str, list[Example]]:
    """
    Return {'train','dev','test','expert'} for ILDC (gated). 'train'/'dev' are the
    union of the single- and multi-petition pools; 'test' is the shared test set;
    'expert' is the 56-doc explanation subset.
    """
    key = _proc_key("ildc", "cjpe", max_samples)
    cached = _load_processed(processed_cache, key)
    if cached is not None:
        return cached

    from datasets import load_dataset
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = _try_load(load_dataset, "Exploration-Lab/IL-TUR", "cjpe", token, cache_dir)

    splits = {
        "train":  _ildc_split(ds, ILDC_TRAIN_SPLITS, max_samples, seed),
        "dev":    _ildc_split(ds, ILDC_DEV_SPLITS, max_samples, seed),
        "test":   _ildc_split(ds, ILDC_TEST_SPLITS, max_samples, seed),
        "expert": _ildc_split(ds, ILDC_EXPERT_SPLITS, None, seed, keep_explanations=True),
    }
    # Loud failure if the split-name mapping did not match the dataset (empty train/dev/test).
    _require_nonempty_splits(splits, ("train", "dev", "test"), available=_dataset_split_names(ds))
    _report_splits("ILDC", splits)
    _save_processed(processed_cache, key, splits)
    return splits


def load_ecthr(cache_dir: str, task: str = "ecthr_a",
               max_samples: int | None = None, seed: int = 13,
               processed_cache: str | None = None) -> dict[str, list[Example]]:
    """
    Return {'train','dev','test'} for ECtHR (LexGLUE). Binary defense-consistent
    label: 1 = at least one violation found (the APPLICANT prevails); 0 = none.
    Subgroup is a single bucket (the task name): ECtHR lacks a defensible
    per-document protected-attribute proxy, so subgroup fairness is reported on
    ILDC; for ECtHR it degenerates to the marginal case (documented).
    """
    key = _proc_key("ecthr", task, max_samples)
    cached = _load_processed(processed_cache, key)
    if cached is not None:
        return cached

    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", task, cache_dir=cache_dir)

    split_map = {"train": "train", "dev": "validation", "test": "test"}
    out: dict[str, list[Example]] = {}
    for our_name, hf_name in split_map.items():
        rows = ds[hf_name]
        examples = []
        for i, row in enumerate(rows):
            text = coerce_text(row["text"])
            if not text.strip():
                continue  # skip empty fact statements
            # defense-consistent: violation found => applicant prevails => favorable (1)
            label = int(len(row["labels"]) > 0)
            examples.append(Example(
                id=f"{task}-{our_name}-{i}",
                text=text,
                label=label,
                subgroup=task,
                ner_spans=[],
                gold_explanation={},
            ))
        out[our_name] = _maybe_subsample(examples, max_samples, seed)
    _require_nonempty_splits(out, ("train", "dev", "test"), available=list(ds.keys()))
    _report_splits(f"ECtHR/{task}", out)
    _save_processed(processed_cache, key, out)
    return out


# --------------------------------------------------------------------------------
# ILDC internals
# --------------------------------------------------------------------------------
def _ildc_split(ds, names, max_samples, seed, keep_explanations=False) -> list[Example]:
    rows: list[Example] = []
    for n in names:
        if n not in ds:
            continue
        for row in ds[n]:
            rows.append(Example(
                id=str(row.get("id")),
                text=coerce_text(row.get("text", "")),
                label=_require_label(row),
                subgroup=_subgroup_of(row),
                ner_spans=_coerce_ner_spans(row.get("spans")),  # names if present; usually empty
                gold_explanation=_structured_explanation(row) if keep_explanations else {},
            ))
    return _maybe_subsample(rows, max_samples, seed)


def _structured_explanation(row) -> dict:
    """
    Preserve the per-expert, per-rank structure of the gold salient sentences for the
    56 expert documents: {expert_i: {rank_j: [sentences]}}. Empty experts/ranks dropped.

    The IL-TUR cjpe schema uses 'rank_1'..'rank_5' (WITH underscore); we read that and
    accept 'rank1'..'rank5' as a fallback so either encoding yields the gold sentences.
    Keys are normalized to the 'rank_j' form in the output.
    """
    out: dict[str, dict] = {}
    for i in range(1, 6):
        e = row.get(f"expert_{i}")
        if not e:
            continue
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except (json.JSONDecodeError, TypeError):
                continue
        ranks = {}
        for r in range(1, 6):
            sents = e.get(f"rank_{r}") or e.get(f"rank{r}") or []
            if sents:
                ranks[f"rank_{r}"] = list(sents)
        if ranks:
            out[f"expert_{i}"] = ranks
    return out


def _subgroup_of(row) -> str:
    """
    Deterministic era bucket from the case id (ILDC ids look like 'YEAR_idx').
    This is a reliability-disaggregation axis (does coverage hold across time
    periods?), NOT a claim about any protected attribute of individuals; the
    paper documents it as such. Only plausible 4-digit years (1800-2099) are
    treated as years; anything else is the 'unk' bucket.
    """
    cid = str(row.get("id", ""))
    prefix = cid[:4]
    if not prefix.isdigit():
        return "unk"
    y = int(prefix)
    if not (1800 <= y <= 2099):
        return "unk"
    if y < 1990:
        return "pre1990"
    if y < 2010:
        return "1990_2010"
    return "post2010"


# --------------------------------------------------------------------------------
# Shared internals
# --------------------------------------------------------------------------------
def _dataset_split_names(ds) -> list:
    """Best-effort list of the dataset's actual split names (for error messages)."""
    try:
        return list(ds.keys())
    except Exception:  # noqa: BLE001
        return []


def _require_nonempty_splits(splits: dict, required, available=None):
    """Raise a loud, actionable error if any required split is empty (split-name mismatch)."""
    empty = [s for s in required if not splits.get(s)]
    if empty:
        avail = f" Dataset's actual split names: {available}." if available else ""
        raise SystemExit(
            f"[data] splits {empty} are EMPTY -- the configured split-name mapping does not "
            f"match the dataset.{avail} Edit the ILDC_*_SPLITS constants in data.py to match."
        )


def _try_load(load_dataset, repo, cfg, token, cache_dir):
    """Try the documented revision first, then fall back; actionable gating message."""
    last = None
    for rev in ("script", "main", None):
        try:
            return load_dataset(repo, cfg, revision=rev, token=token, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001  (we genuinely want to try the next revision)
            last = e
    raise SystemExit(
        f"Could not load {repo}:{cfg}. If this is a gating/auth error, accept the dataset "
        f"terms at https://huggingface.co/datasets/{repo} while logged in, then set HF_TOKEN "
        f"(or run `hf auth login`). Underlying error: {last}"
    )


def _maybe_subsample(rows: list[Example], max_samples: int | None, seed: int) -> list[Example]:
    """Seeded, label-stratified subsample for smoke runs. OFF unless max_samples is set."""
    if max_samples is None or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    pos = [r for r in rows if r.label == 1]
    neg = [r for r in rows if r.label == 0]
    k_pos = max(1, int(round(max_samples * len(pos) / max(1, len(rows)))))
    k_neg = max(1, max_samples - k_pos)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sub = pos[:k_pos] + neg[:k_neg]
    rng.shuffle(sub)
    return sub


def _report_splits(name: str, splits: dict[str, list[Example]]):
    """Print split sizes and label balance -- catches a silent label miscast immediately."""
    print(f"[data] {name} splits:")
    for s, rows in splits.items():
        n = len(rows)
        pos = sum(r.label == 1 for r in rows)
        bal = f"{100.0 * pos / n:.1f}% positive" if n else "empty"
        print(f"        {s:8s} n={n:>6}  {bal}")
        # fail loudly if labels escaped {0,1}
        bad = [r.id for r in rows if r.label not in (0, 1)][:3]
        if bad:
            raise ValueError(f"{name}/{s}: labels outside {{0,1}} for ids {bad}")


# --------------------------------------------------------------------------------
# Processed-split caching (JSONL)
# --------------------------------------------------------------------------------
_PROC_VERSION = "v3"  # bump when the processing logic changes (v3: rank_ fix, text/NER coercion)


def _proc_key(dataset: str, config: str, max_samples: int | None) -> str:
    return f"{dataset}_{config}_{_PROC_VERSION}_ms{max_samples}"


def _save_processed(cache_dir: str | None, key: str, splits: dict[str, list[Example]]):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for split_name, rows in splits.items():
            for r in rows:
                f.write(json.dumps({
                    "split": split_name, "id": r.id, "text": r.text, "label": r.label,
                    "subgroup": r.subgroup, "ner_spans": r.ner_spans,
                    "gold_explanation": r.gold_explanation,
                }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[data] cached processed splits -> {path}")


def _load_processed(cache_dir: str | None, key: str):
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{key}.jsonl")
    if not os.path.exists(path):
        return None
    splits: dict[str, list[Example]] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            splits.setdefault(d["split"], []).append(Example(
                id=d["id"], text=d["text"], label=int(d["label"]),
                subgroup=d["subgroup"],
                ner_spans=[tuple(s) if isinstance(s, list) else s
                           for s in d.get("ner_spans", [])],
                gold_explanation=d.get("gold_explanation", {}),
            ))
    print(f"[data] loaded processed splits from cache {path}")
    return splits


# --------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------
def content_hash(examples: list[Example]) -> str:
    """Fingerprint id + label + a short text digest, so different texts cannot collide."""
    h = hashlib.sha256()
    for e in examples:
        h.update(e.id.encode())
        h.update(b"\x00")
        h.update(str(e.label).encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(e.text.encode()).digest()[:8])
        h.update(b"\x01")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------------
# Self-test: exercises every dependency-free path (coercion, explanation structure,
# subgroup bucketing, stratified subsample, caching round-trip, hashing) against the
# documented schema, WITHOUT requiring the gated download or the `datasets` library.
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # (1) label coercion -- int / string / bool, and loud failure on garbage
    assert coerce_label(1) == 1 and coerce_label(0) == 0
    assert coerce_label("ACCEPTED") == 1 and coerce_label("REJECTED") == 0
    assert coerce_label(" accepted ") == 1 and coerce_label(True) == 1
    for bad in ["MAYBE", 7, None, 2.5]:
        try:
            coerce_label(bad); raise AssertionError(f"{bad!r} should have raised")
        except (ValueError, TypeError):
            pass
    # missing label fails loudly (does NOT default to 0)
    try:
        _require_label({"id": "x", "text": "t"}); raise AssertionError("missing label must raise")
    except KeyError:
        pass

    # (2) text coercion: string, list-of-sentences, None all handled
    assert coerce_text("hello world") == "hello world"
    assert coerce_text(["a", "b", "c"]) == "a b c"
    assert coerce_text(None) == "" and coerce_text([]) == ""

    # (3) structured explanation: schema 'rank_1' (underscore) is read; legacy 'rank1' too
    expert_row_schema = {"id": "1951_10", "label": 1,
                         "expert_1": {"label": 1, "rank_1": ["s1a", "s1b"], "rank_2": ["s2"]},
                         "expert_2": {"label": 1, "rank_1": ["e2s1"]},
                         "expert_3": None}
    se = _structured_explanation(expert_row_schema)
    assert set(se) == {"expert_1", "expert_2"}
    ex = Example("1951_10", "txt", 1, "post2010", [], se)
    assert ex.flat_explanation() == ["s1a", "s1b", "s2", "e2s1"], ex.flat_explanation()
    # legacy 'rank1' (no underscore) still recovered
    legacy = _structured_explanation({"expert_1": {"rank1": ["x"], "rank2": ["y"]}})
    assert legacy == {"expert_1": {"rank_1": ["x"], "rank_2": ["y"]}}

    # (4) NER span coercion: integer labels -> names; dicts and tuples accepted
    spans = _coerce_ner_spans([{"start": 0, "end": 5, "label": 8},   # 8 -> STAT
                               {"start": 10, "end": 15, "label": 9},  # 9 -> PREC
                               (20, 25, "DATE")])
    assert spans == [(0, 5, "STAT"), (10, 15, "PREC"), (20, 25, "DATE")]
    assert _coerce_ner_spans(None) == [] and _coerce_ner_spans([("bad",)]) == []
    assert ner_label_name(10) == "DATE" and ner_label_name("STAT") == "STAT"

    # (5) ECtHR label logic: violation found => favorable (1); none => 0
    assert int(len(["art_3"]) > 0) == 1 and int(len([]) > 0) == 0

    # (6) subgroup bucketing from ILDC-style ids (with plausible-year bound)
    assert _subgroup_of({"id": "1951_10"}) == "pre1990"
    assert _subgroup_of({"id": "1995_3"}) == "1990_2010"
    assert _subgroup_of({"id": "2018_77"}) == "post2010"
    assert _subgroup_of({"id": "abcd_1"}) == "unk"
    assert _subgroup_of({"id": "9999_1"}) == "unk"  # implausible year -> unk

    # (7) loud error when required splits are empty (split-name mismatch)
    try:
        _require_nonempty_splits({"train": [1], "dev": [], "test": [1]}, ("train", "dev", "test"),
                                 available=["foo_train", "bar"])
        raise AssertionError("empty split must raise")
    except SystemExit as e:
        assert "dev" in str(e) and "foo_train" in str(e)

    # (8) stratified subsample is seeded, size-bounded, preserves both classes when present
    rows = ([Example(f"2000_{i}", "t", 1, "post2010") for i in range(80)] +
            [Example(f"2000_{i}", "t", 0, "post2010") for i in range(80, 100)])
    sub = _maybe_subsample(rows, max_samples=40, seed=13)
    assert len(sub) <= 40 and any(r.label == 1 for r in sub) and any(r.label == 0 for r in sub)
    assert _maybe_subsample(rows, None, 13) is rows  # off by default

    # (9) caching round-trip preserves content (including NER tuples + explanations)
    import tempfile
    ex_ner = Example("2001_5", "txt", 1, "post2010", [(0, 5, "STAT")], se)
    with tempfile.TemporaryDirectory() as tmp:
        splits = {"train": rows[:10], "expert": [ex_ner]}
        _save_processed(tmp, "k", splits)
        back = _load_processed(tmp, "k")
        assert len(back["train"]) == 10
        assert back["expert"][0].flat_explanation() == ex.flat_explanation()
        assert back["expert"][0].ner_spans == [(0, 5, "STAT")]

    # (10) content hash distinguishes different texts with same id+label (no collision)
    a = [Example("X", "alpha text", 1, "g")]
    b = [Example("X", "beta text", 1, "g")]
    assert content_hash(a) != content_hash(b)

    print("data.py self-test passed (coercion + missing-label guard + rank_ explanation fix + "
          "text/NER coercion + subgroup + empty-split guard + cache round-trip + collision-free hash)")
