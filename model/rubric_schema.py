"""Shared rubric schema used for synthetic labeling and training.

The schema mirrors the machine-readable rubric specification in
rubric_specification.pdf.
"""

from __future__ import annotations

from typing import Final

RUBRIC_DIMENSIONS: Final[list[str]] = [
    "learning_outcomes",
    "pedagogical_content_knowledge",
    "teaching_learning_strategies",
    "resources_including_ict",
    "assessment_strategies_in_plan",
    "lesson_introduction_rpk",
    "lesson_sequencing",
    "attention_to_all_learners",
    "concept_explanation_examples",
    "lesson_closure",
]

RUBRIC_WEIGHTS: Final[dict[str, float]] = {
    "learning_outcomes": 0.16,
    "pedagogical_content_knowledge": 0.10,
    "teaching_learning_strategies": 0.12,
    "resources_including_ict": 0.08,
    "assessment_strategies_in_plan": 0.14,
    "lesson_introduction_rpk": 0.10,
    "lesson_sequencing": 0.10,
    "attention_to_all_learners": 0.10,
    "concept_explanation_examples": 0.05,
    "lesson_closure": 0.05,
}

RUBRIC_BANDS: Final[dict[str, tuple[int, int]]] = {
    "outstanding": (85, 100),
    "good": (70, 84),
    "minimum_level_of_practice": (50, 69),
    "inadequate": (0, 49),
}


def overall_score(scores: dict[str, int]) -> float:
    """Convert 1-4 criterion scores into a 0-100 weighted score."""
    weighted_mean = sum(scores[key] * RUBRIC_WEIGHTS[key] for key in RUBRIC_DIMENSIONS)
    return (weighted_mean - 1.0) / 3.0 * 100.0


def validate_scores(scores: dict[str, int]) -> None:
    """Raise ValueError if a score map does not match the rubric schema."""
    missing = [key for key in RUBRIC_DIMENSIONS if key not in scores]
    extra = [key for key in scores if key not in RUBRIC_DIMENSIONS]
    if missing or extra:
        raise ValueError(f"rubric score keys mismatch: missing={missing}, extra={extra}")
    for key in RUBRIC_DIMENSIONS:
        if scores[key] < 1 or scores[key] > 4:
            raise ValueError(f"rubric score {key} out of range: {scores[key]}")