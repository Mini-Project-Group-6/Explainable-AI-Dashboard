"""Rubric-guided synthetic lesson plan generator.

Implements the proposal's synthetic data plan (Part D2 + risk register
G2/R1-R2): until the 40-60 real annotated CoE lesson plans arrive, training
runs on template-augmented GES-format plans with *known* rubric scores.

Each plan is generated from a quality profile — one 1-4 score per rubric
dimension — and the text is rendered so that the weaknesses a low
score implies are actually present in the document (vague objectives, no
time allocations, thin assessment, ...). The profile is the label.

Dimensions (order fixed, mirrors rubric_specification.pdf):
    learning_outcomes, pedagogical_content_knowledge, teaching_learning_strategies,
    resources_including_ict, assessment_strategies_in_plan, lesson_introduction_rpk,
    lesson_sequencing, attention_to_all_learners, concept_explanation_examples,
    lesson_closure

Deterministic per seed. Synthetic plans must be labelled as synthetic in any
reported dataset (proposal D2) — the output manifest records this.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from rubric_schema import RUBRIC_DIMENSIONS, validate_scores

# --- Topic bank: content words are shared between objectives, core points and
# --- assessment so that alignment (F13) is controllable per plan.
TOPICS = {
    "Integrated Science": [
        ("Photosynthesis", ["plants", "sunlight", "chlorophyll", "carbon dioxide", "oxygen", "glucose", "leaf"]),
        ("The Water Cycle", ["evaporation", "condensation", "precipitation", "clouds", "rainfall", "rivers"]),
        ("Simple Machines", ["lever", "pulley", "inclined plane", "effort", "load", "fulcrum"]),
        ("States of Matter", ["solid", "liquid", "gas", "melting", "boiling", "particles"]),
    ],
    "Mathematics": [
        ("Fractions", ["numerator", "denominator", "equivalent fractions", "addition", "simplify"]),
        ("Area of Plane Shapes", ["rectangle", "triangle", "square", "length", "breadth", "formula"]),
        ("Percentages", ["percent", "fraction", "decimal", "discount", "profit", "conversion"]),
    ],
    "English Language": [
        ("Nouns and Pronouns", ["noun", "pronoun", "common noun", "proper noun", "sentence", "subject"]),
        ("Comprehension Skills", ["passage", "main idea", "context clues", "inference", "summary"]),
    ],
}

MEASURABLE_VERBS = ["identify", "describe", "explain", "list", "state", "calculate",
                    "demonstrate", "classify", "compare", "solve", "construct", "evaluate"]
VAGUE_VERBS = ["know", "understand", "appreciate", "learn about", "be aware of"]
CRITERIA = ["at least three", "at least two", "correctly", "using the chart provided",
            "with the aid of real objects", "without referring to the textbook"]

ACTIVITY_SENTENCES = [
    "Learners discuss {kw} in small groups and present their findings to the class.",
    "Teacher demonstrates {kw} using TLMs while learners observe and ask questions.",
    "Learners practise solving exercises on {kw} in pairs.",
    "Through question and answer, learners brainstorm what they know about {kw}.",
    "Learners role-play the process of {kw} in front of the class.",
    "Learners draw and label a chart of {kw} in their exercise books.",
    "In groups, learners investigate {kw} and record their observations.",
]

LECTURE_SENTENCES = [
    "Teacher explains {kw} to the class.",
    "Teacher writes notes about {kw} on the board for learners to copy.",
    "Teacher reads the section on {kw} from the textbook.",
]

ASSESSMENT_STEMS = [
    "Explain the role of {kw} in your own words.",
    "List {n} examples of {kw}.",
    "What is meant by {kw}?",
    "Describe how {kw} occurs.",
    "Calculate the answer to the exercise on {kw} on the board.",
]

OFFTOPIC_ASSESSMENT = [
    "Write five sentences about your last holiday.",
    "Copy the notes from the board neatly into your exercise book.",
]

RUN_ON_FILLER = (
    "and this is something that the learners should really try to remember because "
    "it will come up again later in the term when we treat the next topic which is "
    "related to this one in several ways"
)

STAGES = ["INTRODUCTION", "PRESENTATION", "LEARNER ACTIVITIES", "EVALUATION", "CLOSURE"]


@dataclass
class SyntheticPlan:
    plan_id: str
    text: str
    scores: dict[str, int]  # dimension -> 0-4
    subject: str
    topic: str


def _objectives_block(rng, score: int, keywords: list[str], topic: str) -> str:
    if score == 1:
        return "OBJECTIVES:\nTeacher explains the lesson topic."
    lines = ["OBJECTIVES:", "By the end of the lesson, the learner will be able to:"]
    if score >= 4:
        lines.append(f"- {rng.choice(MEASURABLE_VERBS)} a curriculum indicator for {topic} using {rng.choice(CRITERIA)} evidence")
        lines.append(f"- {rng.choice(MEASURABLE_VERBS)} the key idea of {keywords[0]} in one lesson")
        lines.append(f"- {rng.choice(MEASURABLE_VERBS)} {keywords[1 % len(keywords)]} with a clear example")
    elif score == 3:
        lines.append(f"- {rng.choice(MEASURABLE_VERBS)} the main idea of {keywords[0]}")
        lines.append(f"- {rng.choice(MEASURABLE_VERBS)} {keywords[1 % len(keywords)]}")
    else:
        lines.append(f"- {rng.choice(VAGUE_VERBS)} {keywords[0]}")
    return "\n".join(lines)


def _content_block(rng, score: int, topic: str, keywords: list[str]) -> str:
    n_sentences = [1, 2, 4, 6, 8][score]
    lines = ["CORE POINTS:"]
    for i in range(n_sentences):
        kw = keywords[i % len(keywords)]
        lines.append(
            f"{topic} involves {kw}. "
            f"Learners should note that {kw} is a key point of today's lesson.")
    return "\n".join(lines)


def _resources_block(score: int, keywords: list[str]) -> str:
    if score == 1:
        return "RESOURCES:\nNone listed."
    if score == 2:
        return "RESOURCES:\nTextbook, chalkboard, exercise book."
    if score == 3:
        return (
            "RESOURCES:\n"
            f"Flashcards for {keywords[0]}; chart paper for group work; textbook for reference."
        )
    return (
        "RESOURCES:\n"
        f"Locally made flashcards for {keywords[0]}; chart paper for class recording; "
        "real objects or pictures for demonstration; a projector or phone image where ICT is available."
    )


def _differentiation_block(score: int) -> str:
    if score == 1:
        return "DIFFERENTIATION:\nNo reference to differing learner needs."
    if score == 2:
        return "DIFFERENTIATION:\nSlower learners will be helped."
    if score == 3:
        return "DIFFERENTIATION:\nA support task and an extension task are provided for different ability levels."
    return (
        "DIFFERENTIATION:\n"
        "Below-level learners get scaffolded task cards; above-level learners get an extension task; "
        "learners with special educational needs receive targeted support; participation is shared fairly across gender."
    )


def _explanation_block(score: int, keywords: list[str]) -> str:
    if score == 1:
        return "EXPLANATION STRATEGY:\nTeacher talks through the topic."
    if score == 2:
        return f"EXPLANATION STRATEGY:\nExamples are textbook-bound and generic for {keywords[0]}."
    if score == 3:
        return f"EXPLANATION STRATEGY:\nTeacher uses familiar examples to explain {keywords[0]} clearly."
    return (
        "EXPLANATION STRATEGY:\n"
        f"Teacher uses an analogy, diagram, and demonstration from the Ghanaian classroom context to explain {keywords[0]}."
    )


def _intro_block(score: int, topic: str, keywords: list[str]) -> str:
    if score == 1:
        return "INTRODUCTION:\nNo starter or introduction phase."
    if score == 2:
        return f"INTRODUCTION:\nTeacher reviews previous lesson content on {keywords[-1]}."
    if score == 3:
        return (
            "INTRODUCTION:\n"
            f"Teacher reviews learners' prior knowledge on {keywords[-1]} and connects it to {topic}."
        )
    return (
        "INTRODUCTION:\n"
        f"Teacher uses a brief hook, reviews prior knowledge on {keywords[-1]}, connects it to {topic}, and shares the objective with learners."
    )


def _sequencing_block(score: int, with_time: bool) -> str:
    if score == 1:
        return "SEQUENCING:\nNo discernible phase structure."
    if score == 2:
        return "SEQUENCING:\nA phase is missing and timings are absent."
    if score == 3:
        prefix = "SEQUENCING:\nStarter, main, and plenary are present and ordered."
        if with_time:
            prefix += " Timings are allocated."
        return prefix
    prefix = (
        "SEQUENCING:\nStarter, main, and plenary all present with coherent progression; each activity builds on the previous; timings are realistic and balanced."
    )
    if with_time:
        prefix += " Phase timings are shown throughout."
    return prefix


def _closure_block(score: int) -> str:
    if score == 1:
        return "CLOSURE:\nNo plenary or closure."
    if score == 2:
        return "CLOSURE:\nTeacher summary only."
    if score == 3:
        return "CLOSURE:\nPlenary is present and summarizes the lesson."
    return "CLOSURE:\nPlenary consolidates the indicator, learners summarize key points, and closing questions check attainment."


def _activities_block(rng, score: int, keywords: list[str], with_time: bool) -> str:
    n_learner = [0, 1, 2, 3, 4][score]
    n_lecture = [3, 2, 2, 1, 1][score]
    pool = ([rng.choice(ACTIVITY_SENTENCES) for _ in range(n_learner)]
            + [rng.choice(LECTURE_SENTENCES) for _ in range(n_lecture)])
    rng.shuffle(pool)
    lines = ["TEACHER-LEARNER ACTIVITIES:"]
    for i, template in enumerate(pool):
        prefix = f"({rng.choice([10, 15, 20])} minutes) " if with_time else ""
        lines.append(prefix + template.format(kw=keywords[i % len(keywords)]))
    return "\n".join(lines)


def _assessment_block(rng, score: int, keywords: list[str]) -> str:
    n_items = [0, 1, 2, 3, 4][score]
    lines = ["EVALUATION:"]
    if n_items == 0:
        lines.append("Exercise to be given later.")
        return "\n".join(lines)
    for i in range(n_items):
        if score <= 1 and i == 0:
            lines.append(f"{i + 1}. " + rng.choice(OFFTOPIC_ASSESSMENT))
            continue
        stem = rng.choice(ASSESSMENT_STEMS)
        lines.append(f"{i + 1}. " + stem.format(kw=keywords[i % len(keywords)], n=rng.choice([2, 3, 4])))
    return "\n".join(lines)


def _pedagogical_content_block(score: int, keywords: list[str]) -> str:
    if score == 1:
        return "PCK:\nThe plan contains a factual or conceptual error, or content is absent."
    if score == 2:
        return f"PCK:\nContent is broadly correct but thin and generic for {keywords[0]}."
    if score == 3:
        return f"PCK:\nContent is accurate and appropriate to grade level with worked examples for {keywords[0]}."
    return (
        "PCK:\n"
        f"Content is accurate throughout, anticipates common misconceptions for {keywords[0]}, and includes pitched worked examples."
    )


def _apply_language_quality(rng, text: str, score: int) -> str:
    if score >= 3:
        return text
    # Degrade language: append run-on filler to some sentences.
    lines = text.splitlines()
    out = []
    for line in lines:
        if line and not line.isupper() and rng.random() < (0.5 - 0.1 * score):
            line = line.rstrip(".") + " " + RUN_ON_FILLER + "."
        out.append(line)
    return "\n".join(out)


def generate_plan(rng: random.Random, plan_id: str,
                  scores: dict[str, int] | None = None) -> SyntheticPlan:
    """Render one GES-format lesson plan for a (possibly sampled) quality profile."""
    if scores is None:
        # Correlated profile: plans tend to be coherently weak/average/strong,
        # with per-dimension jitter so dimensions stay separable.
        base = rng.choice([1, 2, 2, 3, 3, 4])
        scores = {
            d: min(4, max(1, base + rng.choice([-1, 0, 0, 1])))
            for d in RUBRIC_DIMENSIONS
        }

    validate_scores(scores)

    subject = rng.choice(list(TOPICS))
    topic, keywords = rng.choice(TOPICS[subject])
    kws = list(keywords)
    rng.shuffle(kws)

    with_time = scores["lesson_sequencing"] >= 3
    duration = rng.choice([60, 70, 80])
    # Stated vs allocated consistency degrades with the sequencing score.
    stated = duration if scores["lesson_sequencing"] >= 3 else int(duration * rng.choice([1.5, 0.6]))

    header = "\n".join([
        f"SUBJECT: {subject}",
        f"TOPIC: {topic}",
        f"CLASS: JHS {rng.choice([1, 2, 3])}",
        f"DURATION: {stated} minutes",
        f"CLASS SIZE: {rng.randint(25, 55)}",
    ])

    language_quality = int(round(sum(scores[d] for d in RUBRIC_DIMENSIONS) / len(RUBRIC_DIMENSIONS)))

    parts = [
        header,
        _objectives_block(rng, scores["learning_outcomes"], kws, topic),
        _pedagogical_content_block(scores["pedagogical_content_knowledge"], kws),
        _resources_block(scores["resources_including_ict"], kws),
        _intro_block(scores["lesson_introduction_rpk"], topic, kws),
        _content_block(rng, scores["pedagogical_content_knowledge"], topic, kws),
        _activities_block(rng, scores["teaching_learning_strategies"], kws, with_time),
        _assessment_block(rng, scores["assessment_strategies_in_plan"], kws),
        _differentiation_block(scores["attention_to_all_learners"]),
        _explanation_block(scores["concept_explanation_examples"], kws),
        _sequencing_block(scores["lesson_sequencing"], with_time),
        _closure_block(scores["lesson_closure"]),
    ]
    text = _apply_language_quality(rng, "\n\n".join(parts), language_quality)
    return SyntheticPlan(plan_id=plan_id, text=text, scores=scores,
                         subject=subject, topic=topic)


def generate_dataset(n: int = 200, seed: int = 42) -> list[SyntheticPlan]:
    rng = random.Random(seed)
    return [generate_plan(rng, f"synth_{i:04d}") for i in range(n)]


def write_dataset(plans: list[SyntheticPlan], out_dir: str | Path) -> Path:
    """Write plans as .txt + labels.csv manifest (synthetic flag included)."""
    out_dir = Path(out_dir)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)
    for plan in plans:
        (out_dir / "plans" / f"{plan.plan_id}.txt").write_text(plan.text, encoding="utf-8")
    manifest = out_dir / "labels.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["plan_id", "subject", "topic", "is_synthetic"] + RUBRIC_DIMENSIONS)
        for plan in plans:
            writer.writerow([plan.plan_id, plan.subject, plan.topic, 1]
                            + [plan.scores[d] for d in RUBRIC_DIMENSIONS])
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rubric-labelled synthetic lesson plans")
    parser.add_argument("-n", type=int, default=200, help="number of plans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic_v1", help="output directory")
    args = parser.parse_args()

    dataset = generate_dataset(args.n, args.seed)
    manifest_path = write_dataset(dataset, args.out)
    print(f"Wrote {len(dataset)} plans -> {manifest_path}")
