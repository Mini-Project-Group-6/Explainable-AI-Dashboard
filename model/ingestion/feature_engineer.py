"""18 rubric-aligned features from a segmented lesson plan.

Implements ``docs/FEATURES.md`` (v0.1 draft). Input is a
``LessonPlanText`` from ``ingestion/extract_text.py``; output is an
ordered 18-value feature vector whose names are the exact columns used
by the XGBoost scorer and shown in SHAP waterfall plots.

Design decisions carried over from FEATURES.md:
- F13 uses lemma-overlap cosine similarity (open question 1, option (a))
  because en_core_web_sm ships no word vectors.
- F17 is returned RAW here; z-scoring against the training corpus happens
  in the training pipeline (scoring/train_xgboost.py stores mean/std).
- Missing-section policy: features of a missing section default to 0.
- Everything is deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

from ingestion.extract_text import MAIN_STAGES, LessonPlanText

# Exact SHAP-facing column order — do not reorder without retraining.
FEATURE_NAMES: list[str] = [
    "objective_count",              # F1
    "smart_objective_count",        # F2
    "objective_measurability_ratio",  # F3
    "bloom_remember_prop",          # F4
    "bloom_understand_prop",        # F5
    "bloom_apply_prop",             # F6
    "bloom_analyze_prop",           # F7
    "bloom_evaluate_prop",          # F8
    "bloom_create_prop",            # F9
    "content_activity_ratio",       # F10
    "learner_activity_verb_density",  # F11
    "activity_variety_count",       # F12
    "assessment_alignment_score",   # F13
    "assessment_item_count",        # F14
    "time_allocation_coverage",     # F15
    "time_total_consistency",       # F16
    "sentence_complexity_index",    # F17 (raw; z-scored at training time)
    "readability_flesch",           # F18
    # v0.3 — added so every rubric criterion has at least one feature that
    # measures it. Before these, four criteria were scored by a model that had
    # nothing measuring them, and their SHAP decompositions were not evidence.
    "resource_specificity_count",   # F19
    "rpk_link_score",               # F20
    "differentiation_strategy_count",  # F21
    "closure_quality_score",        # F22
]

# --- Lexicons (FEATURES.md Appendix A; UK + US spellings, stored as lemmas) ---

BLOOM_LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")

BLOOM_VERBS: dict[str, frozenset[str]] = {
    "remember": frozenset({
        "define", "list", "name", "state", "recall", "identify", "label",
        "match", "recognise", "recognize", "select"}),
    "understand": frozenset({
        "explain", "describe", "summarise", "summarize", "classify", "discuss",
        "interpret", "paraphrase", "illustrate", "compare", "outline"}),
    "apply": frozenset({
        "apply", "use", "solve", "demonstrate", "calculate", "complete",
        "show", "implement", "practise", "practice", "sketch"}),
    "analyze": frozenset({
        "analyse", "analyze", "differentiate", "distinguish", "examine",
        "organise", "organize", "contrast", "categorise", "categorize",
        "investigate", "deconstruct"}),
    "evaluate": frozenset({
        "evaluate", "justify", "critique", "judge", "defend", "argue",
        "assess", "appraise", "recommend"}),
    "create": frozenset({
        "create", "design", "construct", "compose", "develop", "formulate",
        "plan", "produce", "invent", "generate"}),
}

MEASURABLE_VERBS: frozenset[str] = frozenset().union(*BLOOM_VERBS.values())

# Non-measurable — excluded from SMART (single-word lemmas checked on tokens,
# multi-word phrases checked on the sentence text).
VAGUE_VERBS = frozenset({"know", "understand", "learn", "appreciate", "grasp"})
VAGUE_PHRASES = ("be aware of", "be familiar with")

INSTRUCTIONAL_VERBS = MEASURABLE_VERBS | VAGUE_VERBS

# F11 — learner-centred activity verbs (lemmas)
LEARNER_ACTIVITY_VERBS = frozenset({
    "discuss", "demonstrate", "present", "solve", "practise", "practice",
    "group", "role-play", "brainstorm", "share", "explore", "measure",
    "draw", "act", "perform", "collaborate", "observe", "experiment"})

# F12 — activity-type taxonomy (type -> trigger keywords, lower-cased substrings)
ACTIVITY_TAXONOMY: dict[str, tuple[str, ...]] = {
    "discussion": ("discuss", "debate", "brainstorm"),
    "group_work": ("group work", "in groups", "pair", "team"),
    "demonstration": ("demonstrat",),
    "practice": ("practice", "practise", "exercise", "drill", "solve"),
    "questioning": ("question", "ask", "quiz"),
    "ict_tlm": ("ict", "computer", "projector", "video", "chart", "tlm",
                "teaching learning material", "flashcard", "model"),
    "role_play": ("role play", "role-play", "drama", "act out"),
    "field_observation": ("field", "observation", "excursion", "nature walk"),
}

# F2 — condition/criterion markers for SMART detection
_CRITERION_WORDS = frozenset({
    "correctly", "accurately", "clearly", "successfully", "appropriately",
    "independently", "fluently"})
_CRITERION_PHRASES = ("at least", "at most", "without", "using", "given",
                      "with the aid of", "within")

# --- v0.3 lexicons (F19-F22) ---

# F19 — concrete, nameable teaching/learning resources. Deliberately specific
# nouns: "TLMs will be used" names nothing and should score 0.
#
# Two things this lexicon must NOT do, both found in review:
#   * match inside another word ("chalk" inside "chalkboard"), which inflated
#     the count for a single named resource;
#   * include words that are ordinarily a lesson's *subject matter* rather than
#     a teaching aid. "leaf", "seed", "stone", "straw" and "bottle" were in the
#     list, so a science plan about photosynthesis scored a resource for
#     mentioning leaves. Those are removed; a genuine leaf specimen is still
#     caught by "specimen" or "real object".
# Matching is whole-word (see _lexicon_pattern), never substring.
RESOURCE_NOUNS: frozenset[str] = frozenset({
    "flashcard", "flash-card", "chart", "poster", "picture", "diagram", "map",
    "textbook", "handout", "worksheet", "exercise book", "chalkboard",
    "whiteboard", "blackboard", "chalk", "marker", "cardboard", "manila",
    "specimen", "real object", "counter", "abacus", "ruler", "measuring tape",
    "projector", "computer", "laptop", "tablet", "video", "radio",
    "recording", "slide", "software", "internet",
})

# ICT subset — the criterion names ICT explicitly, so it is worth a bump.
ICT_NOUNS: frozenset[str] = frozenset({
    "projector", "computer", "laptop", "tablet", "video", "radio",
    "recording", "slide", "software", "internet", "ict",
})

# F20 — cues that the introduction reaches back to prior learning.
RPK_CUES: tuple[str, ...] = (
    "previous knowledge", "prior knowledge", "previous lesson", "last lesson",
    "already know", "already learnt", "already learned", "rpk", "recall",
    "review", "revise", "build on", "learnt before", "learned before",
)

# F21 — distinct differentiation strategies, one bucket each.
#
# Every cue here is matched whole-word. Review found that the bare "sen" cue,
# matched as a substring, fired on "preSENtation" — the canonical header of the
# content section — so this feature returned >= 1 for essentially every plan,
# including plans with no differentiation content whatsoever. "visual" and
# "hearing" had the same problem against "visual aids" and ordinary prose, so
# they are now bound to the impairment they were meant to detect.
DIFFERENTIATION_STRATEGIES: dict[str, tuple[str, ...]] = {
    "support": ("support task", "scaffold", "scaffolded", "struggling",
                "slower learner", "below level", "below-level", "remedial",
                "extra help", "targeted support"),
    "extension": ("extension", "challenge task", "above level", "above-level",
                  "gifted", "finish early", "enrichment", "advanced learner"),
    "sen": ("special educational needs", "sen", "disability", "impairment",
            "hearing impairment", "visual impairment", "inclusive", "inclusion"),
    "grouping": ("mixed ability", "ability group", "pair weaker",
                 "heterogeneous", "grouped by"),
    "equity": ("gender", "girls", "boys equally", "participation is shared",
               "equal opportunity"),
    "language": ("mother tongue", "local language", "multilingual",
                 "language support", "code switch"),
}

# F22 — what a closure has to do to count as one.
CLOSURE_SUMMARY_CUES = ("summar", "recap", "key point", "main point",
                        "consolidat", "review the lesson", "conclude")
CLOSURE_LEARNER_CUES = ("learners summar", "learners state", "learners recall",
                        "ask learners", "learners share", "learners explain",
                        "pupils summar", "learners mention")
CLOSURE_CHECK_CUES = ("check", "confirm", "assess", "objective", "attainment",
                      "question", "verify", "ensure they")

def _lexicon_pattern(terms) -> re.Pattern:
    """Whole-word alternation over a lexicon, longest term first.

    Plain ``term in text`` is wrong for every lexicon in this module and it
    failed silently for two of them: "sen" matched inside "presentation" and
    "chalk" inside "chalkboard", so features intended to measure differentiation
    and resource specificity were partly constants. Word boundaries make a match
    mean what the lexicon says it means. Longest-first ordering keeps
    "measuring tape" from being reported as "measuring".
    """
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in ordered) + r")(?!\w)",
        re.IGNORECASE)


_RESOURCE_RE = _lexicon_pattern(RESOURCE_NOUNS)
_ICT_RE = _lexicon_pattern(ICT_NOUNS)
_DIFFERENTIATION_RES: dict[str, re.Pattern] = {
    bucket: _lexicon_pattern(cues)
    for bucket, cues in DIFFERENTIATION_STRATEGIES.items()
}
_RPK_CUE_RE = _lexicon_pattern(RPK_CUES)

_TIME_RE = re.compile(r"\b(\d+)\s*(?:minutes|mins?)\b", re.IGNORECASE)
_NUMBERED_ITEM_RE = re.compile(r"^\s*(?:\d+[\.\)]|[a-z][\.\)]|[ivx]+[\.\)])\s+",
                               re.IGNORECASE | re.MULTILINE)


def load_nlp():
    """Load the pinned spaCy pipeline (proposal Part D2)."""
    import spacy

    return spacy.load("en_core_web_sm")


class FeatureEngineer:
    """Turns one ``LessonPlanText`` into the 18-value feature vector."""

    def __init__(self, nlp=None):
        self.nlp = nlp or load_nlp()

    # -- public API ---------------------------------------------------------

    def extract_features(self, plan: LessonPlanText) -> dict[str, float]:
        """Ordered dict of the 18 features, keys == FEATURE_NAMES."""
        objectives_doc = self._doc(plan.sections.get("objectives", ""))
        activities_doc = self._doc(plan.sections.get("activities", ""))
        assessment_doc = self._doc(plan.sections.get("assessment", ""))
        content_text = plan.sections.get("content", "")
        full_doc = self._doc(plan.raw_text)

        features: dict[str, float] = {}
        features.update(self._objective_features(objectives_doc))          # F1–F3
        features.update(self._bloom_features(objectives_doc, activities_doc))  # F4–F9
        features["content_activity_ratio"] = self._content_activity_ratio(     # F10
            content_text, plan.sections.get("activities", ""))
        features["learner_activity_verb_density"] = (                      # F11
            self._activity_verb_density(activities_doc))
        features["activity_variety_count"] = self._activity_variety(      # F12
            plan.sections.get("activities", ""))
        features["assessment_alignment_score"] = self._lemma_overlap(     # F13
            objectives_doc, assessment_doc)
        features["assessment_item_count"] = self._assessment_items(       # F14
            plan.sections.get("assessment", ""), assessment_doc)
        features["time_allocation_coverage"] = self._time_coverage(plan)  # F15
        features["time_total_consistency"] = self._time_consistency(plan)  # F16
        features["sentence_complexity_index"] = self._complexity(full_doc)  # F17
        features["readability_flesch"] = self._flesch(full_doc)           # F18
        features["resource_specificity_count"] = self._resource_specificity(  # F19
            plan)
        features["rpk_link_score"] = self._rpk_link(plan, objectives_doc)  # F20
        features["differentiation_strategy_count"] = (                    # F21
            self._differentiation_strategies(plan))
        features["closure_quality_score"] = self._closure_quality(plan)   # F22

        return {name: float(features[name]) for name in FEATURE_NAMES}

    def to_vector(self, plan: LessonPlanText) -> list[float]:
        return list(self.extract_features(plan).values())

    # -- helpers ------------------------------------------------------------

    def _doc(self, text: str):
        return self.nlp(text) if text else self.nlp("")

    @staticmethod
    def _objective_units(doc) -> list:
        """One unit per bullet line or sentence in the objectives section."""
        text = doc.text
        lines = [ln.strip(" \t-•*") for ln in text.splitlines() if ln.strip(" \t-•*")]
        # Bullet-per-line layout if most lines are short; else sentence-split.
        if len(lines) >= 2:
            return lines
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def _objective_features(self, objectives_doc) -> dict[str, float]:
        units = self._objective_units(objectives_doc)
        count = 0
        smart = 0
        for unit in units:
            unit_doc = self.nlp(unit)
            lemmas = {t.lemma_.lower() for t in unit_doc}
            lower = unit.lower()
            is_objective = bool(lemmas & INSTRUCTIONAL_VERBS) or any(
                p in lower for p in VAGUE_PHRASES)
            if not is_objective:
                continue
            count += 1
            if self._is_smart(unit_doc, lemmas, lower):
                smart += 1
        return {
            "objective_count": count,
            "smart_objective_count": smart,
            "objective_measurability_ratio": smart / max(count, 1),
        }

    @staticmethod
    def _is_smart(unit_doc, lemmas: set[str], lower: str) -> bool:
        measurable = bool(lemmas & MEASURABLE_VERBS)
        if not measurable:
            return False
        has_number = any(t.like_num for t in unit_doc)
        has_criterion = (
            has_number
            or bool(lemmas & _CRITERION_WORDS)
            or any(p in lower for p in _CRITERION_PHRASES))
        return has_criterion

    def _bloom_features(self, objectives_doc, activities_doc) -> dict[str, float]:
        counts = Counter()
        for doc in (objectives_doc, activities_doc):
            for token in doc:
                if token.pos_ != "VERB":
                    continue
                lemma = token.lemma_.lower()
                for level, verbs in BLOOM_VERBS.items():
                    if lemma in verbs:
                        counts[level] += 1
                        break
        total = sum(counts.values())
        return {
            f"bloom_{level}_prop": (counts[level] / total if total else 0.0)
            for level in BLOOM_LEVELS
        }

    @staticmethod
    def _content_activity_ratio(content: str, activities: str) -> float:
        # Missing-section policy: either side absent -> 0.
        if not content.strip() or not activities.strip():
            return 0.0
        ratio = len(content.split()) / max(len(activities.split()), 1)
        return min(ratio, 5.0)

    @staticmethod
    def _activity_verb_density(activities_doc) -> float:
        tokens = [t for t in activities_doc if not t.is_space]
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens
                   if t.lemma_.lower() in LEARNER_ACTIVITY_VERBS)
        return hits / len(tokens) * 100.0

    @staticmethod
    def _activity_variety(activities_text: str) -> int:
        lower = activities_text.lower()
        return sum(1 for keywords in ACTIVITY_TAXONOMY.values()
                   if any(k in lower for k in keywords))

    @staticmethod
    def _lemma_overlap(objectives_doc, assessment_doc) -> float:
        """F13 — cosine similarity of content-word lemma counts (tf overlap).

        Lexical stand-in for vector similarity; see FEATURES.md open
        question 1 (en_core_web_sm has no word vectors).
        """
        def bag(doc) -> Counter:
            return Counter(
                t.lemma_.lower() for t in doc
                if t.is_alpha and not t.is_stop)

        a, b = bag(objectives_doc), bag(assessment_doc)
        if not a or not b:
            return 0.0
        dot = sum(count * b[lemma] for lemma, count in a.items())
        norm = math.sqrt(sum(c * c for c in a.values())) * \
            math.sqrt(sum(c * c for c in b.values()))
        return dot / norm if norm else 0.0

    def _assessment_items(self, assessment_text: str, assessment_doc) -> int:
        numbered = len(_NUMBERED_ITEM_RE.findall(assessment_text))
        interrogative = 0
        imperative = 0
        for sent in assessment_doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            if text.endswith("?"):
                interrogative += 1
                continue
            first = next((t for t in sent if not t.is_punct and not t.is_space), None)
            if first is not None and first.lemma_.lower() in MEASURABLE_VERBS:
                imperative += 1
        # Numbered items usually contain the question/instruction itself:
        # take the larger of layout-based and sentence-based counts.
        return max(numbered, interrogative + imperative)

    @staticmethod
    def _time_coverage(plan: LessonPlanText) -> float:
        with_time = sum(
            1 for stage in MAIN_STAGES
            if _TIME_RE.search(plan.sections.get(stage, "")))
        return with_time / len(MAIN_STAGES)

    @staticmethod
    def _time_consistency(plan: LessonPlanText) -> float:
        stated = plan.stated_duration_minutes
        if not stated:
            return 0.0
        total = sum(
            int(m.group(1))
            for stage in MAIN_STAGES
            for m in _TIME_RE.finditer(plan.sections.get(stage, "")))
        if total == 0:
            return 0.0
        relative_error = abs(total - stated) / stated
        return max(0.0, 1.0 - relative_error / 0.5)  # off by >=50% -> 0

    @staticmethod
    def _parse_depth(token) -> int:
        # Compare indices, not identity: spaCy builds a fresh Token object on
        # every .head access, so `head is token` never holds at the root.
        depth = 0
        while token.head.i != token.i:
            token = token.head
            depth += 1
        return depth

    def _complexity(self, full_doc) -> float:
        """F17 raw value: mean sentence length x mean parse-tree depth."""
        sents = [s for s in full_doc.sents
                 if sum(1 for t in s if not t.is_space and not t.is_punct) >= 3]
        if not sents:
            return 0.0
        lengths, depths = [], []
        for sent in sents:
            tokens = [t for t in sent if not t.is_space and not t.is_punct]
            lengths.append(len(tokens))
            depths.append(max((self._parse_depth(t) for t in tokens), default=0))
        return (sum(lengths) / len(lengths)) * (sum(depths) / len(depths))

    # -- v0.3 features (F19-F22) -------------------------------------------

    @staticmethod
    def _section_or_plan(plan: LessonPlanText, section: str) -> str:
        """Section text, falling back to the whole plan if it has no header.

        Plans that never separate out (say) resources still mention them inline,
        and a feature that returned 0 for those would be measuring layout rather
        than practice.
        """
        text = plan.sections.get(section, "")
        return text if text.strip() else plan.raw_text

    def _resource_specificity(self, plan: LessonPlanText) -> float:
        """F19 — how many distinct, concrete resources the plan actually names.

        Counting named nouns rather than section length is the point: "TLMs will
        be provided" names nothing and scores 0, which is the judgement a tutor
        makes.
        """
        text = self._section_or_plan(plan, "resources")
        named = {m.group(0).lower() for m in _RESOURCE_RE.finditer(text)}
        # ICT is called out by the criterion itself, so having any is worth one.
        ict_bonus = 1 if _ICT_RE.search(text) else 0
        return min(10, len(named) + ict_bonus)

    def _rpk_link(self, plan: LessonPlanText, objectives_doc) -> float:
        """F20 — does the opening connect prior knowledge to this lesson?

        Three equal parts: an opening exists, it cues prior learning, and it
        shares vocabulary with what the lesson is about. The third part is what
        separates a real link from the boilerplate "Teacher revises previous
        knowledge" that appears in every weak plan.
        """
        opening = " ".join(
            part for part in (plan.sections.get("rpk", ""),
                              plan.sections.get("introduction", ""))
            if part.strip())
        if not opening.strip():
            return 0.0

        score = 1.0 / 3.0
        if _RPK_CUE_RE.search(opening):
            score += 1.0 / 3.0

        opening_doc = self._doc(opening)
        overlap = self._lemma_overlap(objectives_doc, opening_doc)
        if overlap > 0.05:
            score += 1.0 / 3.0
        return round(score, 4)

    def _differentiation_strategies(self, plan: LessonPlanText) -> float:
        """F21 — distinct kinds of provision for differing learner needs (0-6)."""
        text = self._section_or_plan(plan, "differentiation")
        return sum(1 for pattern in _DIFFERENTIATION_RES.values()
                   if pattern.search(text))

    def _closure_quality(self, plan: LessonPlanText) -> float:
        """F22 — whether the closure does the three things a closure should.

        Present, consolidates, involves the learners, and checks attainment.
        A closure that is only "Teacher summarises" scores partially, which is
        the distinction the rubric draws.
        """
        text = plan.sections.get("closure", "")
        if not text.strip():
            return 0.0
        lower = text.lower()
        parts = [
            1.0,                                                    # present
            any(cue in lower for cue in CLOSURE_SUMMARY_CUES),
            any(cue in lower for cue in CLOSURE_LEARNER_CUES),
            any(cue in lower for cue in CLOSURE_CHECK_CUES),
        ]
        return round(sum(float(p) for p in parts) / len(parts), 4)

    def _flesch(self, full_doc) -> float:
        words = [t for t in full_doc if t.is_alpha]
        sents = [s for s in full_doc.sents if any(t.is_alpha for t in s)]
        if not words or not sents:
            return 0.0
        syllables = sum(_count_syllables(t.text) for t in words)
        return (206.835
                - 1.015 * (len(words) / len(sents))
                - 84.6 * (syllables / len(words)))


def _count_syllables(word: str) -> int:
    """Rule-based English syllable count (no extra dependency)."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    if word.endswith("e") and not word.endswith(("le", "ee")):
        word = word[:-1]
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def features_dataframe(plans: Iterable[LessonPlanText], nlp=None):
    """Feature matrix for a batch of plans (rows follow input order)."""
    import pandas as pd

    engineer = FeatureEngineer(nlp)
    rows = [engineer.extract_features(p) for p in plans]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    import argparse
    import json

    from ingestion.extract_text import extract

    parser = argparse.ArgumentParser(description="Extract the 18-feature vector from a lesson plan")
    parser.add_argument("path", help="Path to a .pdf or .docx lesson plan")
    args = parser.parse_args()

    plan = extract(args.path)
    engineer = FeatureEngineer()
    print(json.dumps(engineer.extract_features(plan), indent=2))
