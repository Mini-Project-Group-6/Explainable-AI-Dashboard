"""Turn negative contributions into concrete revision suggestions.

Rule-based, not generated text. A reviewer can read every sentence a student
teacher is shown and trace it to the rule and the measured value that produced
it; nothing here is sampled from a language model.

Two firing conditions, and both must hold for a SHAP-driven suggestion:

    1. the feature's contribution to that criterion is negative and non-trivial
       (it actually pulled the score down), and
    2. the feature's measured value is outside its healthy range
       (``feature_labels.FeatureLabel.is_weak``).

Condition 2 is the important one. A tree model will happily assign a negative
Shapley value to a feature whose value is perfectly good — that is a statement
about the corpus, not about this plan. Telling a student teacher to fix
something they already did right is exactly the failure that destroys trust in
the tool, which is the outcome this project is measuring. So the model decides
*what matters*; the thresholds decide *what is actionable*; a suggestion needs
both.

Structural features do not measure four of the ten criteria (resources/ICT, RPK
introduction, attention to all learners, closure — see
``feature_labels.uncovered_criteria``). Those get presence rules instead: a
missing section, or the absence of any relevant vocabulary anywhere in the
plan. That keeps every criterion actionable even where SHAP has nothing
trustworthy to say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from model_contract import CRITERION_KEYS, Criterion, criterion
from explainability.feature_labels import FEATURE_LABELS, Direction, FeatureLabel
from explainability.reconcile import Channel, CriterionExplanation, PlanExplanation

#: A contribution below this share of the criterion's total attribution is
#: noise; suggesting a fix for it wastes the student teacher's attention.
MIN_INFLUENCE = 0.02


@dataclass(frozen=True)
class Suggestion:
    """One actionable revision, traceable to the rule that produced it.

    Project objective 22 requires a feature-importance explanation attached to
    *every* AI-generated feedback item, so a SHAP-driven suggestion carries the
    Shapley value and influence share that triggered it, not just prose. Rules
    that are not SHAP-driven (a missing section, absent content) say so through
    ``source`` and leave those fields ``None`` — the honest answer, rather than
    a number with no decomposition behind it.
    """

    id: str                       # stable, e.g. "C01.smart_objective_count.low"
    criterion_id: str
    criterion_key: str
    criterion_label: str
    nts_indicator: str
    message: str                  # imperative, concrete, tutor-reviewed wording
    evidence: str                 # why this fired, with the measured value
    severity: float               # 0-1
    source: str                   # "shap" | "missing_section" | "absent_content"
    feature: Optional[str] = None
    feature_label: Optional[str] = None    # tutor-readable name of the feature
    feature_value: Optional[float] = None  # the measured value, raw units
    shap_value: Optional[float] = None     # signed Shapley value for this criterion
    influence: Optional[float] = None      # share of the criterion's |attribution|
    side: Optional[str] = None             # "low" | "high" — which way the value is wrong

    @property
    def dedupe_key(self) -> str:
        """Identity for de-duplication — the *fix*, not the criterion.

        One weak feature usually drags several criteria down at once: missing
        time allocations lower sequencing, strategies and content knowledge
        together. Keying on the criterion would show the student teacher the
        same instruction three times under three headings, which reads as the
        tool malfunctioning. The advice is about the plan, so it is filed once,
        under the criterion it affected most.
        """
        if self.source == "shap" and self.feature:
            return f"feature:{self.feature}:{self.side}"
        return self.id

    @property
    def priority(self) -> str:
        if self.severity >= 0.66:
            return "high"
        return "medium" if self.severity >= 0.33 else "low"

    @property
    def has_shap_explanation(self) -> bool:
        return self.shap_value is not None

    @property
    def attribution_text(self) -> str:
        """One line stating the SHAP basis, for display under the suggestion."""
        if not self.has_shap_explanation:
            return ("Triggered by a check on what the plan contains, not by a "
                    "feature contribution.")
        return (f"{self.feature_label} contributed {self.shap_value:+.3f} to the "
                f"{self.criterion_label} score "
                f"({self.influence:.0%} of the evidence for this criterion).")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "criterion_id": self.criterion_id,
            "criterion": self.criterion_key,
            "criterion_label": self.criterion_label,
            "nts_indicator": self.nts_indicator,
            "message": self.message,
            "evidence": self.evidence,
            "severity": round(self.severity, 3),
            "priority": self.priority,
            "source": self.source,
            "feature": self.feature,
            "feature_label": self.feature_label,
            "feature_value": (None if self.feature_value is None
                              else round(self.feature_value, 4)),
            "shap_value": (None if self.shap_value is None
                           else round(self.shap_value, 4)),
            "influence": (None if self.influence is None
                          else round(self.influence, 4)),
            "has_shap_explanation": self.has_shap_explanation,
            "attribution_text": self.attribution_text,
        }


# --------------------------------------------------------------------------
# Feature rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureRule:
    """What to say when a feature is below / above its healthy range.

    Templates may use ``{value}`` (formatted for the unit), ``{count}`` (integer)
    and ``{pct}`` (percentage).
    """

    when_low: Optional[str] = None
    when_high: Optional[str] = None


FEATURE_RULES: dict[str, FeatureRule] = {
    "objective_count": FeatureRule(
        when_low=("State at least two learning objectives, each on its own line, "
                  "beginning 'By the end of the lesson the learner will be able to...'."),
        when_high=("Reduce to two to four objectives. {count} is more than one "
                   "lesson period can teach and assess properly."),
    ),
    "smart_objective_count": FeatureRule(
        when_low=("Rewrite the objectives so each has an observable verb and a "
                  "condition or standard — for example 'List at least three uses "
                  "of a lever, correctly' rather than 'Know about levers'."),
    ),
    "objective_measurability_ratio": FeatureRule(
        when_low=("Replace vague verbs (know, understand, appreciate, be aware of) "
                  "with observable ones (list, explain, calculate, demonstrate) so "
                  "attainment can be checked."),
    ),
    "bloom_remember_prop": FeatureRule(
        when_high=("Most tasks in this plan only ask learners to recall. Add at "
                   "least one task that asks them to apply or analyse the idea — "
                   "for example solving a new problem or comparing two cases."),
    ),
    "bloom_understand_prop": FeatureRule(
        when_high=("The plan stays at explaining and describing. Add a task where "
                   "learners use the idea to do something."),
    ),
    "bloom_apply_prop": FeatureRule(
        when_low=("Add a task where learners apply the concept — solve, calculate, "
                  "demonstrate or use it on a fresh example."),
    ),
    "bloom_analyze_prop": FeatureRule(
        when_low=("Add a task that asks learners to compare, sort or examine — for "
                  "example distinguishing two cases and saying why they differ."),
    ),
    "bloom_evaluate_prop": FeatureRule(
        when_low=("Add a short task that asks learners to judge or justify, for "
                  "example 'Which method is better here, and why?'."),
    ),
    "bloom_create_prop": FeatureRule(
        when_low=("Where the topic allows, add a task where learners produce "
                  "something of their own — a design, a plan, or a worked example "
                  "they construct."),
    ),
    "content_activity_ratio": FeatureRule(
        when_low=("The core-points section is thin next to the activities. Write "
                  "out the key teaching points so another teacher could deliver "
                  "this lesson from the plan."),
        when_high=("The plan is weighted towards exposition. Convert part of the "
                   "core-points section into something learners do with the "
                   "content rather than receive."),
    ),
    "learner_activity_verb_density": FeatureRule(
        when_low=("The activities section mostly describes what the teacher does. "
                  "Rewrite at least two steps as things the learners do — discuss, "
                  "solve, demonstrate, sort, present."),
    ),
    "activity_variety_count": FeatureRule(
        when_low=("Only {count} type(s) of activity appear. Add another kind — "
                  "group discussion, a demonstration with a TLM, questioning, or a "
                  "short practice exercise."),
    ),
    "assessment_alignment_score": FeatureRule(
        when_low=("The evaluation questions share little vocabulary with the "
                  "objectives. Rewrite at least one item so it tests, in the same "
                  "terms, what the objectives promised learners would be able to do."),
    ),
    "assessment_item_count": FeatureRule(
        when_low=("Only {count} assessment item(s) are written out. Add items so "
                  "that every objective is checked by at least one question."),
    ),
    "time_allocation_coverage": FeatureRule(
        when_low=("Only {pct} of the lesson stages carry a time in minutes. Add an "
                  "explicit allocation to each stage — introduction, presentation, "
                  "activities, evaluation and closure."),
    ),
    "time_total_consistency": FeatureRule(
        when_low=("The minutes allocated across the stages do not add up to the "
                  "lesson duration stated in the header. Adjust the stage timings "
                  "or correct the stated duration so they agree."),
    ),
    "sentence_complexity_index": FeatureRule(
        when_low=("The plan is written in fragments. Use full sentences so a "
                  "colleague picking it up can follow the intended delivery."),
        when_high=("Sentences run long and heavily nested. Split them — one "
                   "instruction per sentence."),
    ),
    "readability_flesch": FeatureRule(
        when_low=("The plan reads densely. Shorten sentences and prefer plain "
                  "words so the plan is usable in the classroom."),
    ),
    "resource_specificity_count": FeatureRule(
        when_low=("Name the actual materials this lesson needs. '{count} named' "
                  "is not enough to prepare from — list the specific charts, real "
                  "objects, flashcards or ICT you will bring, not just 'TLMs'."),
    ),
    "rpk_link_score": FeatureRule(
        when_low=("Strengthen the opening: state what learners already know that "
                  "this lesson builds on, and use the same terms as the lesson "
                  "content so the link is explicit rather than generic."),
    ),
    "differentiation_strategy_count": FeatureRule(
        when_low=("Say how different learners will be supported. Name at least "
                  "two: a support task for those who struggle, an extension for "
                  "those who finish early, provision for any learner with special "
                  "educational needs."),
    ),
    "closure_quality_score": FeatureRule(
        when_low=("Strengthen the closure. A plenary should do three things: "
                  "consolidate the key points, have the *learners* state them, "
                  "and check the objective was met before they leave."),
    ),
}


# --------------------------------------------------------------------------
# Presence rules — for criteria the structural features cannot measure
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PresenceRule:
    """Fires when none of *keywords* appears anywhere in the plan.

    Keywords are matched **whole-word**. As bare substrings they silently
    misfired: "sen" matched inside "presentation", so any plan with a
    presentation stage suppressed the differentiation suggestion — meaning a
    plan that said nothing about SEN was never told to add it, which is exactly
    the case this rule exists for.
    """

    criterion_key: str
    keywords: tuple[str, ...]
    message: str
    evidence: str
    severity: float = 0.8

    def matches(self, text: str) -> bool:
        from ingestion.feature_engineer import _lexicon_pattern

        return bool(_lexicon_pattern(self.keywords).search(text))


PRESENCE_RULES: tuple[PresenceRule, ...] = (
    PresenceRule(
        criterion_key="resources_including_ict",
        keywords=("tlm", "teaching and learning material", "teaching learning material",
                  "chart", "flashcard", "flash card", "projector", "ict", "computer",
                  "real object", "picture", "diagram", "model", "resource", "textbook"),
        message=("Name the teaching and learning materials this lesson needs — the "
                 "specific charts, real objects, flashcards or ICT you will use, not "
                 "just 'TLMs'."),
        evidence="No teaching or learning resource is named anywhere in the plan.",
    ),
    PresenceRule(
        criterion_key="attention_to_all_learners",
        keywords=("mixed ability", "special educational needs", "sen",
                  "differentiation", "differentiated", "support task",
                  "extension", "struggling", "slower learner", "gifted",
                  "gender", "inclusive", "all learners", "below level",
                  "below-level", "above level", "above-level", "scaffold",
                  "scaffolded"),
        message=("Say how learners with different needs will be supported. Add a "
                 "support task for learners who struggle and an extension task for "
                 "those who finish early, and note any learner needing targeted help."),
        evidence="The plan does not mention differentiation, SEN support or inclusion.",
    ),
    PresenceRule(
        criterion_key="lesson_introduction_rpk",
        keywords=("rpk", "relevant previous knowledge", "previous knowledge",
                  "prior knowledge", "review", "recap", "introduction", "starter",
                  "set induction"),
        message=("Open with relevant previous knowledge: state what learners already "
                 "know that this lesson builds on, and how you will draw it out."),
        evidence="No introduction or previous-knowledge stage was found.",
    ),
    PresenceRule(
        criterion_key="lesson_closure",
        keywords=("closure", "conclusion", "summary", "summarise", "summarize",
                  "plenary", "recap", "consolidat"),
        message=("Add a closure stage: how learners will summarise what they learned "
                 "and how you will check the objective was met before they leave."),
        evidence="No closure or plenary stage was found.",
    ),
)

#: Missing canonical section -> the criterion it damages.
SECTION_TO_CRITERION: dict[str, str] = {
    "objectives": "learning_outcomes",
    "content": "pedagogical_content_knowledge",
    "activities": "teaching_learning_strategies",
    "assessment": "assessment_strategies_in_plan",
    "introduction": "lesson_introduction_rpk",
    "rpk": "lesson_introduction_rpk",
    "resources": "resources_including_ict",
    "differentiation": "attention_to_all_learners",
    "closure": "lesson_closure",
}

SECTION_MESSAGES: dict[str, str] = {
    "objectives": ("The plan has no objectives section. Add one stating what "
                   "learners will be able to do by the end of the lesson."),
    "content": ("The plan has no core-points section. Write out the content to be "
                "taught, not only the activities."),
    "activities": ("The plan has no activities section. Set out, step by step, what "
                   "the teacher and the learners each do."),
    "assessment": ("The plan has no evaluation section. Add the questions or tasks "
                   "you will use to check learning."),
    "introduction": ("The plan has no introduction. Add a starter that links to "
                     "learners' previous knowledge."),
    "closure": ("The plan has no closure. Add a plenary that consolidates the "
                "lesson and checks attainment."),
    "rpk": ("The plan does not state learners' relevant previous knowledge. Add "
            "what they already know that this lesson builds on."),
    "resources": ("The plan has no resources section. List the teaching and "
                  "learning materials the lesson needs."),
    "differentiation": ("The plan says nothing about differing learner needs. Add "
                        "how you will support learners who struggle and stretch "
                        "those who finish early."),
}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _format(template: str, value: float, label: FeatureLabel) -> str:
    return template.format(
        value=f"{value:.2f}",
        count=int(round(value)),
        pct=f"{value:.0%}" if value <= 1.0 else f"{value:.0f}",
    )


def _shortfall(label: FeatureLabel, value: float) -> tuple[float, str]:
    """(0-1 distance outside the healthy range, "low"|"high"|"")."""
    low, high = label.healthy_range
    if low is not None and value < low:
        scale = abs(low) if low else 1.0
        return min(1.0, (low - value) / scale), "low"
    if high is not None and value > high:
        scale = abs(high) if high else 1.0
        return min(1.0, (value - high) / scale), "high"
    return 0.0, ""


def _suggestion_from_feature(target: Criterion, feature_name: str, value: float,
                             influence: float, shap_value: float
                             ) -> Optional[Suggestion]:
    label = FEATURE_LABELS[feature_name]
    shortfall, side = _shortfall(label, value)
    if not side:
        return None                      # measured value is fine — say nothing

    rule = FEATURE_RULES.get(feature_name)
    template = (rule.when_low if side == "low" else rule.when_high) if rule else None
    if not template:
        return None                      # no defensible advice for this side

    severity = min(1.0, 0.5 * shortfall + 0.5 * min(1.0, influence * 4))
    return Suggestion(
        id=f"{target.id}.{feature_name}.{side}",
        criterion_id=target.id,
        criterion_key=target.key,
        criterion_label=target.label,
        nts_indicator=target.nts_indicator,
        message=_format(template, value, label),
        evidence=(f"{label.label} is {_format('{value}', value, label)} "
                  f"({label.unit}), outside the expected range, and it lowered the "
                  f"{target.label} score."),
        severity=severity,
        source="shap",
        feature=feature_name,
        feature_label=label.label,
        feature_value=float(value),
        shap_value=float(shap_value),
        influence=float(influence),
        side=side,
    )


def suggestions_for_criterion(explanation: CriterionExplanation,
                              feature_values: Mapping[str, float],
                              *, min_influence: float = MIN_INFLUENCE,
                              limit: int = 2) -> list[Suggestion]:
    """SHAP-driven suggestions for one criterion, strongest first."""
    found: list[Suggestion] = []
    for contribution in explanation.evidence:
        if contribution.channel is not Channel.STRUCTURAL:
            continue
        if contribution.raises_score or contribution.influence < min_influence:
            continue
        name = contribution.feature
        if name is None or name not in feature_values:
            continue
        suggestion = _suggestion_from_feature(
            explanation.criterion, name, float(feature_values[name]),
            contribution.influence, contribution.value)
        if suggestion is not None:
            found.append(suggestion)

    found.sort(key=lambda s: s.severity, reverse=True)
    return found[:limit]


def presence_suggestions(raw_text: str,
                         missing_sections: Sequence[str] = ()) -> list[Suggestion]:
    """Suggestions from what the plan does not contain at all."""
    found: list[Suggestion] = []
    seen: set[str] = set()

    for section in missing_sections:
        key = SECTION_TO_CRITERION.get(section)
        if key is None or section not in SECTION_MESSAGES:
            continue
        target = criterion(key)
        found.append(Suggestion(
            id=f"{target.id}.section.{section}",
            criterion_id=target.id,
            criterion_key=key,
            criterion_label=target.label,
            nts_indicator=target.nts_indicator,
            message=SECTION_MESSAGES[section],
            evidence=f"No '{section}' section was found in the uploaded plan.",
            severity=0.9,
            source="missing_section",
        ))
        seen.add(key)

    for rule in PRESENCE_RULES:
        if rule.criterion_key in seen:
            continue                     # already flagged by a missing section
        if rule.matches(raw_text):
            continue
        target = criterion(rule.criterion_key)
        found.append(Suggestion(
            id=f"{target.id}.absent.{rule.criterion_key}",
            criterion_id=target.id,
            criterion_key=rule.criterion_key,
            criterion_label=target.label,
            nts_indicator=target.nts_indicator,
            message=rule.message,
            evidence=rule.evidence,
            severity=rule.severity,
            source="absent_content",
        ))
    return found


#: Ceiling on how much of the panel "you did not include X" items may take.
#: Without it a plan the segmenter failed on produces up to nine missing-section
#: suggestions at severity 0.9 across eight criteria, which fill max_total
#: outright and push every SHAP-driven item off the list — so the tutor is told
#: the tool could not read the document, then given eight items blaming the
#: student teacher for sections the tool simply did not find.
PRESENCE_SHARE_OF_PANEL: float = 0.5


def revision_suggestions(explanation: PlanExplanation,
                         feature_values: Mapping[str, float],
                         *, raw_text: str = "",
                         missing_sections: Sequence[str] = (),
                         format_recognised: bool = True,
                         min_influence: float = MIN_INFLUENCE,
                         max_per_criterion: int = 2,
                         max_total: int = 8) -> tuple[Suggestion, ...]:
    """The full ranked suggestion list for one plan.

    Args:
        explanation: from ``reconcile.reconcile_plan``.
        feature_values: the RAW extracted feature dict (F17 not z-scored).
        raw_text: the plan text, for the presence rules.
        missing_sections: from ``LessonPlanText.missing_sections``.
        format_recognised: from ``LessonPlanText.format_recognised``. When
            False the segmenter probably failed, so "you did not include X" is
            not a claim we can make — presence rules are suppressed entirely and
            only measured weaknesses are reported.
        max_per_criterion: cap so one weak criterion cannot fill the panel.
        max_total: cap so a weak plan gets a workable list, not 30 items.
    """
    collected: list[Suggestion] = []
    if format_recognised:
        collected += presence_suggestions(raw_text, missing_sections)

    for criterion_explanation in explanation.criteria:
        # A criterion with no trustworthy evidence channel gets presence rules
        # only — its SHAP decomposition is not about this criterion.
        if not criterion_explanation.is_explainable:
            continue
        collected += suggestions_for_criterion(
            criterion_explanation, feature_values,
            min_influence=min_influence, limit=max_per_criterion)

    # One fix appears once, filed under the criterion it hurt most (see
    # Suggestion.dedupe_key). Ties on severity are broken by the *absolute*
    # Shapley value, not by influence: influence is a share within one
    # criterion, so a feature that is the only contributor scores 1.0 for every
    # criterion and cannot distinguish between them. Absolute magnitude can.
    def rank(suggestion: Suggestion) -> tuple[float, float, float]:
        return (suggestion.severity,
                abs(suggestion.shap_value or 0.0),
                suggestion.influence or 0.0)

    deduped: dict[str, Suggestion] = {}
    for suggestion in collected:
        existing = deduped.get(suggestion.dedupe_key)
        if existing is None or rank(suggestion) > rank(existing):
            deduped[suggestion.dedupe_key] = suggestion

    order = {key: index for index, key in enumerate(CRITERION_KEYS)}
    ranked = sorted(deduped.values(),
                    key=lambda s: (-s.severity, order[s.criterion_key], s.id))

    # Keep the list spread across criteria rather than stacked on the worst one,
    # and keep "you did not include X" from crowding out the measured findings:
    # presence rules all carry severity 0.9, well above any SHAP-driven item, so
    # unchecked they would sort to the top and fill the panel.
    presence_budget = max(1, int(max_total * PRESENCE_SHARE_OF_PANEL))
    per_criterion: dict[str, int] = {}
    presence_used = 0
    final: list[Suggestion] = []
    for suggestion in ranked:
        if suggestion.source != "shap":
            if presence_used >= presence_budget:
                continue
            presence_used += 1
        count = per_criterion.get(suggestion.criterion_key, 0)
        if count >= max_per_criterion:
            continue
        per_criterion[suggestion.criterion_key] = count + 1
        final.append(suggestion)
        if len(final) >= max_total:
            break
    return tuple(final)
