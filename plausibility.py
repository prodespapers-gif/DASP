"""
plausibility.py  —  CONTRIBUTION 1: legally-grounded plausibility constraints on edits.

A counterfactual that flips the predicted outcome is useless to a defense attorney unless
the *edit it represents* is something the attorney could actually argue. Polyjuice (Wu et
al. 2021) and any free-form generator will happily edit immutable facts (a co-defendant's
confession already in the record, an undisputed date of death). Those edits are fluent,
outcome-flipping, and legally meaningless.

This module defines a TAXONOMY of edit types grounded in legal procedure and a FILTER that
scores whether a candidate edit is legally plausible. "Counterfactual reachability" in
selective.py is defined over the *plausible* edits only -- which is what makes the
abstention signal legally meaningful rather than lexical.

It also provides the bridge to the generator: EDIT_TYPE_TO_POLYJUICE maps each MUTABLE
legal edit type to the Polyjuice control codes (Wu et al. 2021, Table 1) that tend to
produce it, so counterfactual.py can STEER generation toward edits the filter will accept
instead of generating freely and discarding most candidates. Immutable types map to no
codes (they should never be generated). This closes the gap between what a linguistic
generator produces and what a legal filter wants.

Design notes for reproducibility:
- The taxonomy is explicit and inspectable (no hidden heuristics).
- The filter has two tiers: (a) a hard rule layer that blocks edits touching immutable
  spans or classified as an immutable edit type, and (b) a learned plausibility scorer
  over InLegalBERT embeddings. Either tier can run alone (ablation: rules-only vs.
  rules+learned).
- A lightweight, transparent keyword classifier (classify_edit_type) gives every edit a
  type WITHOUT the learned tier, so the taxonomy is actually used in rules-only mode; a
  stronger classifier can be injected via classify_fn.
- Immutable-span detection keys off IL-TUR L-NER categories (the 12-type legal NER label
  set: Statute, Precedent, Date, Case Number, Court, Judge, Authority, ...) plus regex
  detectors for the dates/quantities that constitute undisputed physical facts.

Reference: Wu, Ribeiro, Heer, Weld. Polyjuice: Generating Counterfactuals for Explaining,
Evaluating, and Improving Models. ACL-IJCNLP 2021 (control codes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable


# --------------------------------------------------------------------------------------
# 1. The edit taxonomy
# --------------------------------------------------------------------------------------
class EditType(str, Enum):
    # --- MUTABLE: things a defense attorney can legitimately contest/introduce ---
    WITNESS_CREDIBILITY      = "witness_credibility"        # contest reliability of testimony
    EVIDENCE_ADMISSIBILITY   = "evidence_admissibility"     # challenge how evidence was obtained
    MITIGATING_CIRCUMSTANCE  = "mitigating_circumstance"    # introduce mitigation
    PROCEDURAL_ERROR         = "procedural_error"           # raise a procedural/Charter defect
    ALTERNATIVE_INTERPRET    = "alternative_interpretation"  # reinterpret an ambiguous fact
    # --- IMMUTABLE: things already fixed in the record; editing them is not an argument ---
    PRIOR_CONVICTION         = "prior_conviction"           # immutable
    CODEFENDANT_CONFESSION   = "codefendant_confession"     # immutable
    UNDISPUTED_PHYSICAL_FACT = "undisputed_physical_fact"   # immutable (date, location, death)
    STATUTORY_TEXT           = "statutory_text"             # immutable (law text itself)
    UNKNOWN                  = "unknown"

    @property
    def is_mutable(self) -> bool:
        return self in _MUTABLE


_MUTABLE = {
    EditType.WITNESS_CREDIBILITY, EditType.EVIDENCE_ADMISSIBILITY,
    EditType.MITIGATING_CIRCUMSTANCE, EditType.PROCEDURAL_ERROR,
    EditType.ALTERNATIVE_INTERPRET,
}


# --------------------------------------------------------------------------------------
# 1b. Bridge to the generator: legal edit type -> Polyjuice control codes (Wu et al. 2021)
# --------------------------------------------------------------------------------------
# Polyjuice's eight control codes are LINGUISTIC; our edit types are LEGAL. They do not
# correspond automatically, so we map each MUTABLE type to the codes most likely to realize
# it. counterfactual.py uses this to request targeted edits (and to restrict edit locations
# to mutable regions), raising the yield of plausible flipping counterfactuals. Immutable
# types intentionally map to no codes: they must never be generated.
POLYJUICE_CONTROL_CODES = (
    "negation", "quantifier", "lexical", "resemantic",
    "insert", "delete", "restructure", "shuffle",
)

EDIT_TYPE_TO_POLYJUICE: dict[EditType, list[str]] = {
    EditType.WITNESS_CREDIBILITY:     ["negation", "lexical"],      # negate/weaken testimony
    EditType.EVIDENCE_ADMISSIBILITY:  ["delete", "negation"],       # remove/negate evidence
    EditType.MITIGATING_CIRCUMSTANCE: ["insert", "quantifier"],     # add mitigating facts
    EditType.PROCEDURAL_ERROR:        ["insert", "restructure"],    # introduce a defect clause
    EditType.ALTERNATIVE_INTERPRET:   ["resemantic", "lexical"],    # reinterpret, same structure
}


def polyjuice_codes_for(edit_type: EditType) -> list[str]:
    """Control codes to request from Polyjuice for a given (mutable) edit type; [] if immutable."""
    return list(EDIT_TYPE_TO_POLYJUICE.get(edit_type, []))


# --------------------------------------------------------------------------------------
# 1c. Lightweight keyword classifier (so the taxonomy is used without the learned tier)
# --------------------------------------------------------------------------------------
# Transparent, inspectable surface cues for each mutable edit type. This is deliberately
# simple (a stronger model can be injected via classify_fn); its job is to ensure rules-only
# mode actually assigns edit types rather than treating every edit as UNKNOWN.
_MUTABLE_KEYWORDS: dict[EditType, tuple[str, ...]] = {
    EditType.WITNESS_CREDIBILITY:     ("witness", "testimony", "credib", "uncorroborat",
                                       "unreliable", "recant", "perjur"),
    EditType.EVIDENCE_ADMISSIBILITY:  ("evidence", "admissib", "obtained", "search", "seizure",
                                       "warrant", "exclud", "chain of custody"),
    EditType.MITIGATING_CIRCUMSTANCE: ("mitigat", "remorse", "first offence", "first offense",
                                       "provocation", "duress", "rehabilitat"),
    EditType.PROCEDURAL_ERROR:        ("procedur", "natural justice", "charter", "due process",
                                       "jurisdiction", "irregular", "ultra vires"),
    EditType.ALTERNATIVE_INTERPRET:   ("interpret", "ambigu", "alternativ", "construe",
                                       "could mean", "reasonable doubt"),
}


def classify_edit_type(edited_text: str) -> EditType:
    """
    Heuristically classify an edit by surface cues in the EDITED text. Returns the first
    matching mutable type in _MUTABLE_KEYWORDS order (so when several cues co-occur the
    precedence is deterministic and documented), else UNKNOWN. Transparent by design; a
    stronger classifier can be injected into LegalPlausibilityFilter via classify_fn, and
    when an EditCandidate carries an explicit edit_type that always takes priority.
    """
    t = edited_text.lower()
    for etype, kws in _MUTABLE_KEYWORDS.items():
        if any(k in t for k in kws):
            return etype
    return EditType.UNKNOWN


# --------------------------------------------------------------------------------------
# 2. Immutable-span detection (hard rule tier)
# --------------------------------------------------------------------------------------
# Keys off (a) legal-NER spans passed in from data.py, and (b) regex detectors for the
# quantities/dates that constitute undisputed physical facts. Conservative by design:
# when in doubt, mark immutable (a false "immutable" only costs us a candidate CF; a false
# "mutable" would let an implausible CF through, which is the failure we must avoid).

# Dates: a day-month-year phrase OR a bare plausible year (1800-2099). The year bound stops
# the previous behavior of matching ANY 4-digit number (docket pages, quantities, amounts).
_DATE_RE = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+(?:1[89]\d{2}|20\d{2})|(?:1[89]\d{2}|20\d{2}))\b"
)
# Quantities/amounts that are undisputed physical facts (weights, money, percentages, ages).
_QUANTITY_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:kg|g|grams?|rupees?|rs\.?|inr|%|years?)\b", re.I)

# IL-TUR L-NER categories (12-type legal NER set) that anchor immutable content. Witness is
# deliberately EXCLUDED: a witness existing is fixed, but witness CREDIBILITY is contestable
# (a mutable edit type), so blocking all WITNESS spans would forbid the defense's core move.
# Accepts common spellings/abbreviations of the same categories for robustness across NER
# label conventions (full names, short forms, integer labels are normalized by the caller).
_IMMUTABLE_NER = {
    "STATUTE", "STAT", "PROVISION",          # statutory text
    "PRECEDENT", "PREC",                     # cited precedent
    "DATE",                                  # fixed dates
    "CASE_NUMBER", "CASENO", "CASENUMBER",   # case/docket numbers
    "COURT",                                 # the deciding court
    "JUDGE",                                 # the judge(s)
    "AUTHORITY",                             # cited authority
}


@dataclass
class Span:
    start: int
    end: int
    label: str  # NER label or detector tag

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"Span end ({self.end}) < start ({self.start})")


def _normalize_ner_label(label) -> str:
    """Normalize an NER label (string or int-as-str) to an upper-case comparable token."""
    return str(label).strip().upper().replace(" ", "_")


def detect_immutable_spans(text: str, ner_spans: list[Span] | None = None) -> list[Span]:
    """
    Return character spans that an edit must NOT touch: immutable legal-NER spans plus
    detected dates and quantities. Spans are de-duplicated (same start/end/label kept once).
    """
    spans: list[Span] = []
    if ner_spans:
        spans += [s for s in ner_spans if _normalize_ner_label(s.label) in _IMMUTABLE_NER]
    for m in _DATE_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "DATE"))
    for m in _QUANTITY_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "QUANTITY"))
    # de-duplicate while preserving order
    seen, unique = set(), []
    for s in spans:
        key = (s.start, s.end, s.label)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def _overlaps(a_start: int, a_end: int, spans: list[Span]) -> bool:
    """
    True if [a_start, a_end) overlaps any span [s.start, s.end) (half-open intervals; edit
    offsets are start-inclusive, end-exclusive, matching counterfactual.py's generator).
    A zero-width edit (a_start == a_end, a pure insertion) is treated as overlapping a span
    iff the insertion point lies strictly INSIDE the span (touching a boundary is allowed),
    so an insertion adjacent to an immutable span is not spuriously blocked.
    """
    if a_start == a_end:  # pure insertion at a point
        return any(s.start < a_start < s.end for s in spans)
    return any(not (a_end <= s.start or a_start >= s.end) for s in spans)


# --------------------------------------------------------------------------------------
# 3. The plausibility filter
# --------------------------------------------------------------------------------------
@dataclass
class EditCandidate:
    original: str
    edited: str
    edit_start: int          # char offset of the changed region in `original` (inclusive)
    edit_end: int            # char offset one past the changed region (exclusive)
    edit_type: EditType = EditType.UNKNOWN


@dataclass
class PlausibilityResult:
    plausible: bool
    edit_type: EditType
    hard_pass: bool          # passed the immutable-span + immutable-type rules
    learned_score: float     # [0,1] from the learned scorer (0.5 if scorer disabled)
    reason: str


class LegalPlausibilityFilter:
    """
    Two-tier filter.
      Tier A (hard): reject if the edited region overlaps an immutable span, or if the
                     classified edit type is immutable.
      Tier B (learned): score plausibility with an InLegalBERT-based classifier. The
                     scorer is injected (a callable str,str -> float) so this file stays
                     framework-light and unit-testable; model.py supplies the real one.

    Edit-type classification: if a candidate carries an explicit edit_type, it is used;
    otherwise classify_fn (if injected) is tried, then the built-in keyword classifier.
    This guarantees the taxonomy is applied even in rules-only mode.

    UNKNOWN policy (`admit_unknown`): an edit whose type cannot be determined is, by
    default, admitted by the hard tier (admit_unknown=True) so the pipeline's default
    rules-only path is unchanged and edits without obvious surface cues are not dropped.
    The unconditional safety win is independent of this flag: immutable SPANS and immutable
    edit TYPES are always rejected. Set admit_unknown=False for a stricter regime that
    refuses any edit it cannot classify (the natural choice once the generator labels edit
    types, which makes UNKNOWN rare); when the learned tier is active, UNKNOWN edits are
    deferred to the scorer regardless of this flag.

    Ablations supported directly:
      - rules_only=True            -> Tier A alone.
      - classify_fn=None           -> use the built-in keyword classifier.
      - scorer=None                -> Tier B disabled (learned_score fixed at 0.5).
      - admit_unknown=False        -> rules-only refuses unclassifiable edits (stricter).
    """

    def __init__(
        self,
        scorer: Callable[[str, str], float] | None = None,
        classify_fn: Callable[[str, str], EditType] | None = None,
        threshold: float = 0.5,
        rules_only: bool = False,
        admit_unknown: bool = True,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.scorer = scorer
        self.classify_fn = classify_fn
        self.threshold = threshold
        self.rules_only = rules_only
        self.admit_unknown = admit_unknown

    def classify_edit(self, original: str, edited: str) -> EditType:
        """Injected classifier first, then the transparent built-in keyword classifier."""
        if self.classify_fn is not None:
            return self.classify_fn(original, edited)
        return classify_edit_type(edited)

    def assess(self, cand: EditCandidate, immutable_spans: list[Span]) -> PlausibilityResult:
        # Tier A.1 -- does the edit physically touch an immutable span?
        if _overlaps(cand.edit_start, cand.edit_end, immutable_spans):
            return PlausibilityResult(False, cand.edit_type, False, 0.0,
                                      "edit overlaps an immutable span")
        # Tier A.2 -- classify edit type; immutable types are rejected outright.
        etype = (cand.edit_type if cand.edit_type != EditType.UNKNOWN
                 else self.classify_edit(cand.original, cand.edited))
        if etype != EditType.UNKNOWN and not etype.is_mutable:
            return PlausibilityResult(False, etype, False, 0.0,
                                      f"edit type '{etype.value}' is legally immutable")

        # Resolve the UNKNOWN policy explicitly.
        if etype == EditType.UNKNOWN and not self.admit_unknown:
            if self.rules_only or self.scorer is None:
                # No way to justify an unclassifiable edit without a scorer -> reject.
                return PlausibilityResult(False, etype, True, 0.5,
                                          "edit type unknown; not admitted by rules alone")
            # else: defer to the learned tier below.

        if self.rules_only or self.scorer is None:
            return PlausibilityResult(True, etype, True, 0.5, "passed hard rules (scorer off)")

        # Tier B -- learned plausibility.
        score = float(self.scorer(cand.original, cand.edited))
        passed = score >= self.threshold
        return PlausibilityResult(passed, etype, True, score,
                                  "learned plausibility " + ("pass" if passed else "fail"))

    def filter(self, cands: list[EditCandidate],
               immutable_spans: list[Span]) -> list[tuple[EditCandidate, PlausibilityResult]]:
        out = []
        for c in cands:
            r = self.assess(c, immutable_spans)
            if r.plausible:
                out.append((c, r))
        return out


# --------------------------------------------------------------------------------------
# Self-test (CPU, no model needed)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    text = "The appellant was convicted on 5th May 2001 based on the witness statement of PW-1."

    # ---- immutable-span detection: date is immutable ----
    spans = detect_immutable_spans(text)
    assert any(s.label == "DATE" for s in spans), "date should be immutable"

    # ---- tightened date regex: arbitrary 4-digit numbers are NOT dates ----
    assert not any(s.label == "DATE" for s in detect_immutable_spans("seized 5000 rupees and 1234 grams"))
    assert any(s.label == "QUANTITY" for s in detect_immutable_spans("seized 5000 rupees and 1234 grams"))

    filt = LegalPlausibilityFilter(rules_only=True)

    # ---- edit touching the date (immutable span) -> rejected ----
    c_bad = EditCandidate(text, text.replace("2001", "2010"),
                          edit_start=text.index("2001"), edit_end=text.index("2001") + 4)
    assert filt.assess(c_bad, spans).plausible is False

    # ---- edit contesting the witness statement (mutable) -> allowed; auto-classified ----
    region = "witness statement of PW-1"
    s = text.index(region)
    c_ok = EditCandidate(text, text.replace(region, "disputed and uncorroborated " + region),
                         edit_start=s, edit_end=s + len(region),
                         edit_type=EditType.WITNESS_CREDIBILITY)
    r_ok = filt.assess(c_ok, spans)
    assert r_ok.plausible is True and r_ok.edit_type == EditType.WITNESS_CREDIBILITY

    # ---- built-in classifier assigns a type when none is given ----
    # (classify on a clean edit phrase; when multiple cues co-occur, first match by the
    #  documented _MUTABLE_KEYWORDS order wins -- see the precedence test below)
    assert classify_edit_type("evidence was obtained without a warrant") == EditType.EVIDENCE_ADMISSIBILITY
    assert classify_edit_type("the testimony was uncorroborated") == EditType.WITNESS_CREDIBILITY
    assert classify_edit_type("an unrelated sentence") == EditType.UNKNOWN
    # precedence is deterministic when several cues co-occur (witness checked before evidence)
    assert classify_edit_type("the witness gave evidence") == EditType.WITNESS_CREDIBILITY

    # ---- immutable edit TYPE is rejected even away from any span ----
    c_imm = EditCandidate("x", "the statutory text of section 302 reads differently",
                          edit_start=0, edit_end=1, edit_type=EditType.STATUTORY_TEXT)
    assert filt.assess(c_imm, []).plausible is False

    # ---- UNKNOWN policy: default (admit_unknown=True) admits unclassifiable edits ----
    c_unk = EditCandidate("the sky", "the sky is a different colour", edit_start=0, edit_end=7)
    assert filt.assess(c_unk, []).plausible is True, "default admits unknown (backward-compatible)"
    # ...strict mode refuses them (the stricter ablation / future default)
    filt_strict_rules = LegalPlausibilityFilter(rules_only=True, admit_unknown=False)
    assert filt_strict_rules.assess(c_unk, []).plausible is False, "strict mode rejects unknown"

    # ---- learned tier: UNKNOWN deferred to scorer; scorer decides ----
    filt_learned = LegalPlausibilityFilter(scorer=lambda a, b: 0.9, threshold=0.5)
    assert filt_learned.assess(c_unk, []).plausible is True   # scorer 0.9 >= 0.5
    filt_strict = LegalPlausibilityFilter(scorer=lambda a, b: 0.1, threshold=0.5)
    assert filt_strict.assess(c_unk, []).plausible is False   # scorer 0.1 < 0.5

    # ---- insertion adjacent to an immutable span is allowed (boundary, not interior) ----
    date_s = text.index("5th May 2001")
    c_adj = EditCandidate(text, text, edit_start=date_s, edit_end=date_s)  # insert at span start
    # touching the boundary is permitted; only strictly-interior insertions are blocked
    assert _overlaps(date_s, date_s, spans) is False

    # ---- Polyjuice control-code mapping: every mutable type has codes; immutable has none ----
    for et in _MUTABLE:
        codes = polyjuice_codes_for(et)
        assert codes and all(c in POLYJUICE_CONTROL_CODES for c in codes)
    assert polyjuice_codes_for(EditType.STATUTORY_TEXT) == []

    # ---- threshold validation ----
    try:
        LegalPlausibilityFilter(threshold=1.5); raise AssertionError("should reject threshold>1")
    except ValueError:
        pass

    # ---- Span validation ----
    try:
        Span(10, 5, "X"); raise AssertionError("should reject end<start")
    except ValueError:
        pass

    print("plausibility.py self-test passed "
          "(immutable spans + tightened dates + keyword classifier + UNKNOWN policy + "
          "immutable-type rejection + Polyjuice control-code mapping verified)")
