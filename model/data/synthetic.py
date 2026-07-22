"""Rubric-guided synthetic lesson plan generator.

Implements the proposal's synthetic data plan (Part D2 + risk register
G2/R1-R2): until the 40-60 real annotated CoE lesson plans arrive, training
runs on template-augmented GES-format plans with *known* rubric scores.

Each plan is generated from a quality profile — one 0-4 score per GES/NCTE
rubric dimension — and the text is rendered so that the weaknesses a low
score implies are actually present in the document (vague objectives, no
time allocations, thin assessment, ...). The profile is the label.

Dimensions (order fixed, mirrors docs/FEATURES.md):
    D1 objectives, D2 content, D3 methods, D4 assessment,
    D5 language, D6 time_management

Deterministic per seed. Synthetic plans must be labelled as synthetic in any
reported dataset (proposal D2) — the output manifest records this.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

RUBRIC_DIMENSIONS = [
    "objectives", "content", "methods", "assessment", "language", "time_management",
]

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


def _objectives_block(rng, score: int, keywords: list[str]) -> str:
    n_objectives = [1, 2, 3, 3, 4][score]
    n_smart = [0, 0, 1, 2, n_objectives][score]
    lines = ["OBJECTIVES:", "By the end of the lesson, the learner will be able to:"]
    for i in range(n_objectives):
        kw = keywords[i % len(keywords)]
        if i < n_smart:
            verb = rng.choice(MEASURABLE_VERBS)
            crit = rng.choice(CRITERIA)
            lines.append(f"- {verb} {crit} facts about {kw}")
        else:
            verb = rng.choice(VAGUE_VERBS if score < 3 else MEASURABLE_VERBS)
            lines.append(f"- {verb} {kw}")
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
        base = rng.choice([0, 1, 2, 2, 3, 3, 4])
        scores = {
            d: min(4, max(0, base + rng.choice([-1, 0, 0, 1])))
            for d in RUBRIC_DIMENSIONS
        }

    subject = rng.choice(list(TOPICS))
    topic, keywords = rng.choice(TOPICS[subject])
    kws = list(keywords)
    rng.shuffle(kws)

    time_score = scores["time_management"]
    with_time = time_score >= 2
    duration = rng.choice([60, 70, 80])
    # Stated vs allocated consistency degrades with the time score.
    stated = duration if time_score >= 3 else int(duration * rng.choice([1.5, 0.6]))

    header = "\n".join([
        f"SUBJECT: {subject}",
        f"TOPIC: {topic}",
        f"CLASS: JHS {rng.choice([1, 2, 3])}",
        f"DURATION: {stated} minutes",
        f"CLASS SIZE: {rng.randint(25, 55)}",
    ])

    intro_time = f"({rng.choice([5, 10])} minutes) " if with_time else ""
    introduction = ("INTRODUCTION:\n"
                    + intro_time
                    + f"Teacher reviews learners' relevant previous knowledge on {kws[-1]} "
                      "through question and answer.")
    rpk = f"R.P.K.:\nLearners have already been taught {kws[-1]} in the previous lesson."

    closure_time = f"({rng.choice([5, 10])} minutes) " if with_time else ""
    closure = ("CLOSURE:\n"
               + closure_time
               + "Teacher summarises the main points of the lesson and learners "
                 "ask questions for clarification.")

    parts = [
        header,
        rpk,
        _objectives_block(rng, scores["objectives"], kws),
        introduction,
        _content_block(rng, scores["content"], topic, kws),
        _activities_block(rng, scores["methods"], kws, with_time),
        _assessment_block(rng, scores["assessment"], kws),
        closure,
    ]
    text = _apply_language_quality(rng, "\n\n".join(parts), scores["language"])
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
