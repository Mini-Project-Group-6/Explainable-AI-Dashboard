"""Tests for the unrecognised-format guard.

The section header keyword lists are still a draft (FEATURES.md open question
2). A plan laid out in a format they do not anticipate has most of its features
default to 0 and would be scored as a very poor lesson. A student teacher must
not be told their planning is inadequate when the tool simply could not read
the document.

Run from the ``model/`` directory:
    python -m unittest tests.test_format_guard -v
"""

from __future__ import annotations

import unittest

from ingestion.extract_text import (
    EXPECTED_SECTIONS,
    SECTION_RECOGNITION_FLOOR,
    LessonPlanText,
    segment_sections,
)

WELL_FORMED = (
    "SUBJECT: Integrated Science\nDURATION: 60 minutes\n"
    "OBJECTIVES:\nLearners will list three parts of a leaf.\n"
    "RPK:\nLearners know that plants have leaves.\n"
    "RESOURCES:\nFlashcards and a wall chart.\n"
    "INTRODUCTION:\nTeacher reviews previous knowledge.\n"
    "CORE POINTS:\nA leaf has a blade, petiole and veins.\n"
    "LEARNER ACTIVITIES:\nLearners discuss in groups.\n"
    "EVALUATION:\n1. Name two parts of a leaf.\n"
    "DIFFERENTIATION:\nA support task is provided.\n"
    "CLOSURE:\nLearners summarise the key points.\n"
)

# Real content, no headings the segmenter knows — the failure mode that matters.
UNHEADED = (
    "Today the class will look at leaves. I want them to be able to name the "
    "parts. We will start by talking about what they saw on the way to school. "
    "Then I will show them a real leaf and we will name the blade and the "
    "veins together. In groups they will draw one. At the end I will ask two "
    "questions to see if they understood.\n"
)


def plan_from(text: str) -> LessonPlanText:
    sections, missing = segment_sections(text)
    return LessonPlanText(source_path="<test>", raw_text=text,
                          sections=sections, missing_sections=missing)


class TestFormatGuard(unittest.TestCase):
    def test_a_well_formed_plan_is_recognised(self):
        plan = plan_from(WELL_FORMED)
        self.assertTrue(plan.format_recognised)
        self.assertIsNone(plan.format_warning)
        self.assertGreaterEqual(plan.sections_found, SECTION_RECOGNITION_FLOOR)

    def test_an_unheaded_plan_is_flagged_not_scored_harshly(self):
        plan = plan_from(UNHEADED)
        self.assertFalse(plan.format_recognised)
        self.assertIsNotNone(plan.format_warning)

    def test_the_warning_blames_the_tool_not_the_teacher(self):
        warning = plan_from(UNHEADED).format_warning
        self.assertIn("this tool does not yet read", warning)
        self.assertNotIn("poor", warning.lower())

    def test_sections_found_is_consistent_with_missing(self):
        for text in (WELL_FORMED, UNHEADED):
            plan = plan_from(text)
            self.assertEqual(
                plan.sections_found,
                len(EXPECTED_SECTIONS) - len(plan.missing_sections))

    def test_empty_document(self):
        plan = plan_from("")
        self.assertEqual(plan.sections_found, 0)
        self.assertFalse(plan.format_recognised)

    def test_the_new_v03_sections_are_segmented(self):
        sections, _ = segment_sections(WELL_FORMED)
        self.assertIn("flashcard", sections["resources"].lower())
        self.assertIn("support task", sections["differentiation"].lower())

    def test_floor_is_a_minority_of_expected_sections(self):
        # The guard must fire only on genuine parse failure, not on a plan that
        # merely omits a few sections — omissions are the rubric's business.
        self.assertLess(SECTION_RECOGNITION_FLOOR, len(EXPECTED_SECTIONS) / 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
