"""Tutor-readable labels for the 18 structural features.

A SHAP waterfall plot is only an *explanation* if a teacher educator can read
the axis. ``objective_measurability_ratio`` is a column name; "Share of
objectives that are measurable" is an explanation. This module is the mapping
between the two, plus the link from each feature to the rubric criterion and
NTS indicator it speaks to.

Feature -> criterion mapping
----------------------------
``docs/FEATURES.md`` v0.1 mapped the features onto a **six**-dimension
"GES/NCTE" rubric (D1-D6). That is superseded: the live rubric is the ten
plan-assessable criteria in ``rubric_schema.py``, and the body is GTEC, not
GES/NCTE. The map below is authoritative.

Re-mapping onto the ten criteria originally left four of them with no feature
that measured anything about them — resources/ICT, the RPK introduction,
attention to all learners, and closure. A regressor was still trained for each,
so each still emitted a score, but its SHAP decomposition was over features
measuring something else and was not evidence about the criterion.

**Closed in v0.3** by F19-F22, one feature per affected criterion. The
machinery that handled the gap is deliberately kept rather than deleted:
``coverage_report()``, ``uncovered_criteria()`` and ``tabular_confidence()``
still compute from the live map, so if a future criterion is added without a
feature the reconciler degrades honestly again instead of silently presenting a
decomposition it should not. ``uncovered_criteria()`` returning empty is a
result, not a reason to remove the check.

Thresholds
----------
``healthy_range`` gates *revision suggestions only* (see
``explainability/suggestions.py``). It never touches scoring. The values are
pedagogical heuristics for a JHS-length lesson, meant to be tuned by S4/S5,
not learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional

from model_contract import (
    CRITERION_KEYS,
    FEATURE_ORDER,
    ArtifactContractError,
    Criterion,
    criterion,
)


class Direction(str, Enum):
    """Which way a feature has to move for the plan to get better."""

    UP = "higher_is_better"
    DOWN = "lower_is_better"
    BAND = "optimal_range"      # both too little and too much are weaknesses


Bound = Optional[float]


@dataclass(frozen=True)
class FeatureLabel:
    """Everything the UI needs to render one feature comprehensibly."""

    name: str                        # contract column name
    code: str                        # F1..F18, matches docs/FEATURES.md
    label: str                       # short axis label for SHAP plots
    meaning: str                     # one sentence, tutor-facing
    unit: str
    direction: Direction
    healthy_range: tuple[Bound, Bound]   # (min, max); None = unbounded that side
    primary_criterion: str           # criterion key this feature mainly evidences
    secondary_criteria: tuple[str, ...] = ()
    plan_section: str = ""           # section of the plan it is read from

    @property
    def criterion(self) -> Criterion:
        return criterion(self.primary_criterion)

    @property
    def nts_indicator(self) -> str:
        return self.criterion.nts_indicator

    def is_weak(self, value: float) -> bool:
        """True if *value* falls outside the healthy range for this feature.

        Takes the RAW extractor value (F17 raw, not z-scored).
        """
        low, high = self.healthy_range
        if low is not None and value < low:
            return True
        if high is not None and value > high:
            return True
        return False


FEATURE_LABELS: Final[dict[str, FeatureLabel]] = {
    f.name: f for f in (
        # --- Objectives ---------------------------------------------------
        FeatureLabel(
            name="objective_count",
            code="F1",
            label="Number of objectives",
            meaning="How many learning objectives the plan states.",
            unit="objectives",
            direction=Direction.BAND,
            healthy_range=(2, 4),
            primary_criterion="learning_outcomes",
            plan_section="objectives",
        ),
        FeatureLabel(
            name="smart_objective_count",
            code="F2",
            label="Measurable (SMART) objectives",
            meaning=("Objectives written with an observable verb and a condition "
                     "or criterion, so attainment can actually be checked."),
            unit="objectives",
            direction=Direction.UP,
            healthy_range=(2, None),
            primary_criterion="learning_outcomes",
            plan_section="objectives",
        ),
        FeatureLabel(
            name="objective_measurability_ratio",
            code="F3",
            label="Share of objectives that are measurable",
            meaning=("Proportion of the stated objectives that are measurable — "
                     "separates 'few but well written' from 'many but vague'."),
            unit="proportion (0-1)",
            direction=Direction.UP,
            healthy_range=(0.6, None),
            primary_criterion="learning_outcomes",
            plan_section="objectives",
        ),

        # --- Cognitive demand (Bloom) ------------------------------------
        FeatureLabel(
            name="bloom_remember_prop",
            code="F4",
            label="Recall-level tasks",
            meaning=("Share of instructional verbs at Bloom's lowest level "
                     "(list, name, state). A plan built only on recall is "
                     "under-pitched."),
            unit="proportion of verbs (0-1)",
            direction=Direction.DOWN,
            healthy_range=(None, 0.40),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("learning_outcomes",),
            plan_section="objectives + activities",
        ),
        FeatureLabel(
            name="bloom_understand_prop",
            code="F5",
            label="Comprehension-level tasks",
            meaning=("Share of verbs asking learners to explain, describe or "
                     "summarise."),
            unit="proportion of verbs (0-1)",
            direction=Direction.BAND,
            healthy_range=(None, 0.60),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("learning_outcomes",),
            plan_section="objectives + activities",
        ),
        FeatureLabel(
            name="bloom_apply_prop",
            code="F6",
            label="Application-level tasks",
            meaning="Share of verbs asking learners to use or apply what they learn.",
            unit="proportion of verbs (0-1)",
            direction=Direction.UP,
            healthy_range=(0.10, None),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="objectives + activities",
        ),
        FeatureLabel(
            name="bloom_analyze_prop",
            code="F7",
            label="Analysis-level tasks",
            meaning="Share of verbs asking learners to compare, examine or distinguish.",
            unit="proportion of verbs (0-1)",
            direction=Direction.UP,
            healthy_range=(0.05, None),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="objectives + activities",
        ),
        FeatureLabel(
            name="bloom_evaluate_prop",
            code="F8",
            label="Evaluation-level tasks",
            meaning="Share of verbs asking learners to judge, justify or assess.",
            unit="proportion of verbs (0-1)",
            direction=Direction.UP,
            healthy_range=(0.05, None),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="objectives + activities",
        ),
        FeatureLabel(
            name="bloom_create_prop",
            code="F9",
            label="Creation-level tasks",
            meaning="Share of verbs asking learners to design, construct or produce.",
            unit="proportion of verbs (0-1)",
            direction=Direction.UP,
            healthy_range=(0.05, None),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="objectives + activities",
        ),

        # --- Content & methods -------------------------------------------
        FeatureLabel(
            name="content_activity_ratio",
            code="F10",
            label="Content-to-activity balance",
            meaning=("Length of the core-points section against the learner-activity "
                     "section. Near 1 is balanced; high means lecture-heavy, low "
                     "means the content is too thin to teach from."),
            unit="ratio of word counts",
            direction=Direction.BAND,
            healthy_range=(0.5, 2.0),
            primary_criterion="pedagogical_content_knowledge",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="content + activities",
        ),
        FeatureLabel(
            name="learner_activity_verb_density",
            code="F11",
            label="Learner-centred activity language",
            meaning=("How often the activities section asks learners to *do* "
                     "something (discuss, demonstrate, solve) rather than "
                     "receive it."),
            unit="verbs per 100 words",
            direction=Direction.UP,
            healthy_range=(2.0, None),
            primary_criterion="teaching_learning_strategies",
            plan_section="activities",
        ),
        FeatureLabel(
            name="activity_variety_count",
            code="F12",
            label="Variety of activity types",
            meaning=("How many distinct activity types appear (discussion, group "
                     "work, demonstration, practice, questioning, ICT/TLM use, "
                     "role play, field work), out of 8."),
            unit="activity types (0-8)",
            direction=Direction.UP,
            healthy_range=(3, None),
            primary_criterion="teaching_learning_strategies",
            secondary_criteria=("resources_including_ict",),
            plan_section="activities",
        ),

        # --- Assessment ---------------------------------------------------
        FeatureLabel(
            name="assessment_alignment_score",
            code="F13",
            label="Assessment matches the objectives",
            meaning=("Vocabulary overlap between what the objectives promise and "
                     "what the evaluation section actually tests."),
            unit="similarity (0-1)",
            direction=Direction.UP,
            healthy_range=(0.30, None),
            primary_criterion="assessment_strategies_in_plan",
            secondary_criteria=("learning_outcomes",),
            plan_section="objectives + assessment",
        ),
        FeatureLabel(
            name="assessment_item_count",
            code="F14",
            label="Number of assessment items",
            meaning=("Questions, numbered items or instructions in the evaluation "
                     "section."),
            unit="items",
            direction=Direction.UP,
            healthy_range=(3, None),
            primary_criterion="assessment_strategies_in_plan",
            plan_section="assessment",
        ),

        # --- Sequencing & timing ------------------------------------------
        FeatureLabel(
            name="time_allocation_coverage",
            code="F15",
            label="Lesson stages with a time allocation",
            meaning=("Share of the five main stages (introduction, presentation, "
                     "activities, evaluation, closure) that carry an explicit "
                     "time in minutes."),
            unit="proportion of stages (0-1)",
            direction=Direction.UP,
            healthy_range=(0.80, None),
            primary_criterion="lesson_sequencing",
            secondary_criteria=("lesson_closure",),
            plan_section="all stages",
        ),
        FeatureLabel(
            name="time_total_consistency",
            code="F16",
            label="Stage times add up to the lesson length",
            meaning=("Whether the minutes allocated across the stages match the "
                     "duration stated in the plan header."),
            unit="agreement (0-1)",
            direction=Direction.UP,
            healthy_range=(0.80, None),
            primary_criterion="lesson_sequencing",
            plan_section="header + all stages",
        ),

        # --- Written clarity ----------------------------------------------
        FeatureLabel(
            name="sentence_complexity_index",
            code="F17",
            label="Sentence complexity",
            meaning=("Mean sentence length weighted by grammatical depth. Flags "
                     "both note-fragment writing and run-on sentences."),
            unit="index (raw; z-scored for the model)",
            direction=Direction.BAND,
            healthy_range=(20, 100),
            primary_criterion="concept_explanation_examples",
            plan_section="whole plan",
        ),
        FeatureLabel(
            name="readability_flesch",
            code="F18",
            label="Readability",
            meaning=("Flesch Reading Ease over the whole plan — a professional-clarity "
                     "signal for a document another teacher may have to pick up."),
            unit="Flesch score (0-100)",
            direction=Direction.BAND,
            healthy_range=(30, 70),
            primary_criterion="concept_explanation_examples",
            plan_section="whole plan",
        ),

        # --- v0.3: one feature for each previously unmeasured criterion ---
        FeatureLabel(
            name="resource_specificity_count",
            code="F19",
            label="Named teaching and learning materials",
            meaning=("How many specific resources the plan names — charts, real "
                     "objects, flashcards, ICT. Saying 'TLMs will be used' names "
                     "nothing and counts zero."),
            unit="named resources",
            direction=Direction.UP,
            healthy_range=(2, None),
            primary_criterion="resources_including_ict",
            secondary_criteria=("teaching_learning_strategies",),
            plan_section="resources",
        ),
        FeatureLabel(
            name="rpk_link_score",
            code="F20",
            label="Link to previous knowledge",
            meaning=("Whether the opening exists, cues what learners already know, "
                     "and connects it to this lesson's content rather than saying "
                     "so generically."),
            unit="score (0-1)",
            direction=Direction.UP,
            healthy_range=(0.6, None),
            primary_criterion="lesson_introduction_rpk",
            secondary_criteria=("lesson_sequencing",),
            plan_section="rpk + introduction",
        ),
        FeatureLabel(
            name="differentiation_strategy_count",
            code="F21",
            label="Ways different learners are supported",
            meaning=("Distinct kinds of provision named: support tasks, extension "
                     "work, SEN provision, ability grouping, gender equity, "
                     "language support."),
            unit="strategies (0-6)",
            direction=Direction.UP,
            healthy_range=(2, None),
            primary_criterion="attention_to_all_learners",
            plan_section="differentiation",
        ),
        FeatureLabel(
            name="closure_quality_score",
            code="F22",
            label="Quality of the lesson closure",
            meaning=("Whether the closure is present, consolidates the key points, "
                     "involves the learners in doing so, and checks the objective "
                     "was met."),
            unit="score (0-1)",
            direction=Direction.UP,
            healthy_range=(0.5, None),
            primary_criterion="lesson_closure",
            plan_section="closure",
        ),
    )
}


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------

def label_for(name: str) -> FeatureLabel:
    """Label record for a contract feature name."""
    try:
        return FEATURE_LABELS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} has no label. Every contract feature must have one — "
            f"add it to FEATURE_LABELS.") from None


def display_name(name: str) -> str:
    """Short tutor-readable label, for a SHAP axis tick or a table row."""
    return label_for(name).label


def axis_labels() -> list[str]:
    """Display labels in contract order — pass straight to a SHAP plot."""
    return [FEATURE_LABELS[name].label for name in FEATURE_ORDER]


def features_for(criterion_key: str, include_secondary: bool = True) -> tuple[str, ...]:
    """Feature names evidencing a criterion, in contract order."""
    criterion(criterion_key)  # validates
    return tuple(
        name for name in FEATURE_ORDER
        if FEATURE_LABELS[name].primary_criterion == criterion_key
        or (include_secondary
            and criterion_key in FEATURE_LABELS[name].secondary_criteria)
    )


def coverage_report() -> dict[str, dict[str, tuple[str, ...]]]:
    """criterion key -> {"primary": (...), "secondary": (...)} feature names."""
    return {
        key: {
            "primary": tuple(n for n in FEATURE_ORDER
                             if FEATURE_LABELS[n].primary_criterion == key),
            "secondary": tuple(n for n in FEATURE_ORDER
                               if key in FEATURE_LABELS[n].secondary_criteria),
        }
        for key in CRITERION_KEYS
    }


def uncovered_criteria() -> tuple[str, ...]:
    """Criteria with no feature that directly measures them.

    The tabular model still scores these; its SHAP decomposition for them is
    not evidence about the criterion and must not be presented as such.
    """
    report = coverage_report()
    return tuple(k for k in CRITERION_KEYS if not report[k]["primary"])


def tabular_confidence(criterion_key: str) -> float:
    """How far the structural features can speak for a criterion, in [0, 1].

    A blunt instrument on purpose: the share of evidence the tabular channel
    can legitimately claim, used by ``reconcile.py`` to weight channels and by
    the UI to caveat a panel. 0.0 means "structural features say nothing here".
    """
    report = coverage_report()[criterion_key]
    primary, secondary = len(report["primary"]), len(report["secondary"])
    if not primary:
        return 0.25 * min(secondary, 2) / 2.0   # indirect signal only
    return min(1.0, 0.5 + 0.25 * min(primary, 2))


# --------------------------------------------------------------------------
# Import-time self-check
# --------------------------------------------------------------------------

def _self_check() -> None:
    labelled = tuple(FEATURE_LABELS)
    if labelled != FEATURE_ORDER:
        missing = [n for n in FEATURE_ORDER if n not in FEATURE_LABELS]
        extra = [n for n in FEATURE_LABELS if n not in FEATURE_ORDER]
        raise ArtifactContractError(
            "FEATURE_LABELS does not cover the contract feature set exactly: "
            f"missing={missing}, unexpected={extra}. Every feature shown in a "
            "SHAP plot needs a readable label.")

    codes = [f.code for f in FEATURE_LABELS.values()]
    if len(set(codes)) != len(codes):
        raise ArtifactContractError(f"duplicate feature codes: {codes}")

    for feature in FEATURE_LABELS.values():
        criterion(feature.primary_criterion)
        for key in feature.secondary_criteria:
            criterion(key)
        low, high = feature.healthy_range
        if low is not None and high is not None and low > high:
            raise ArtifactContractError(
                f"{feature.name}: healthy_range {feature.healthy_range} is inverted")


_self_check()


if __name__ == "__main__":
    from model_contract import criterion as _criterion

    print(f"{'':4} {'feature':32} {'label':44} criterion / NTS")
    print("-" * 118)
    for feature_name in FEATURE_ORDER:
        f = FEATURE_LABELS[feature_name]
        print(f"{f.code:4} {f.name:32} {f.label:44} "
              f"{f.criterion.id} {f.criterion.label} (NTS {f.nts_indicator})")

    print("\nCriterion coverage by structural features")
    print("-" * 118)
    for key, cover in coverage_report().items():
        c = _criterion(key)
        primary = ", ".join(FEATURE_LABELS[n].code for n in cover["primary"]) or "-"
        secondary = ", ".join(FEATURE_LABELS[n].code for n in cover["secondary"]) or "-"
        flag = "  <-- NO DIRECT FEATURE" if not cover["primary"] else ""
        print(f"{c.id} {c.label:36} primary: {primary:28} secondary: {secondary:12}"
              f" conf={tabular_confidence(key):.2f}{flag}")

    gaps = uncovered_criteria()
    if gaps:
        print(f"\n{len(gaps)} criteria have no direct structural evidence: "
              f"{', '.join(_criterion(k).id for k in gaps)}")
        print("Their SHAP decompositions must be caveated or withheld — see "
              "explainability/reconcile.py.")
