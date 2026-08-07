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

Two cautions when reporting metrics from this generator
-------------------------------------------------------
1. **No feature may be an exact function of its label.** Where a rendered count
   maps one-to-one onto a score, the feature derived from it becomes a copy of
   the label and cross-validation reports a meaningless perfect score. Item and
   activity counts are therefore jittered into overlapping ranges. If you add a
   block, check the same way: featurise a sample and cross-tabulate the feature
   against its label.
2. **Quality profiles are deliberately correlated** — real plans that are weak
   in one area are usually weak in others, and the generator reproduces that.
   The side effect is that a criterion with no feature measuring it can still be
   predicted from features measuring *other* criteria. Held-out QWK for
   resources/ICT, RPK introduction, attention to all learners and closure is
   therefore optimistic on synthetic data and should not be read as evidence
   that those criteria are measured. See docs/FEATURES.md, "The coverage gap".
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from rubric_schema import RUBRIC_DIMENSIONS, overall_score, validate_scores

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


# Resource phrasings by kind. Sampling distinct kinds is what makes F19 a
# measure of specificity rather than of sentence length.
RESOURCE_POOL: dict[str, list[str]] = {
    "print": ["flashcards for {kw}", "a wall chart showing {kw}",
              "the class textbook", "worksheets on {kw}",
              "a labelled diagram of {kw}"],
    "realia": ["real objects collected from the school compound",
               "specimens of {kw}", "a simple model of {kw}"],
    "board": ["chalkboard and chalk", "a whiteboard marker for key terms"],
    "ict": ["a projector where power is available", "a phone image of {kw}",
            "a short video clip on {kw}"],
    "manipulative": ["counters for the group task", "a ruler and measuring tape",
                     "cardboard cut-outs prepared beforehand"],
}

DIFFERENTIATION_POOL: dict[str, list[str]] = {
    "support": ["A support task with scaffolded steps is prepared for learners "
                "who struggle.",
                "Slower learners receive targeted support during the group work."],
    "extension": ["An extension task is ready for learners who finish early.",
                  "A challenge question is provided for the most able."],
    "sen": ["Learners with special educational needs are seated in front and "
            "given adapted materials.",
            "Provision is made for a learner with a hearing impairment."],
    "grouping": ["Groups are mixed ability so learners support one another.",
                 "Learners are grouped by ability for the practice task."],
    "equity": ["Participation is shared fairly between girls and boys.",
               "Questions are directed equally across the class."],
    "language": ["Key terms are explained in the local language where needed.",
                 "Learners may first discuss in their mother tongue."],
}


def _graded(rng, score: int) -> bool:
    """Include a quality component with probability rising in *score*, jittered.

    The jitter is the point. Deterministic inclusion makes the derived feature a
    copy of the label — see the module docstring, caution 1.
    """
    return rng.random() < (score - 1 + rng.uniform(-0.6, 0.6)) / 3.0


def _resources_block(rng, score: int, keywords: list[str]) -> str:
    n_kinds = max(0, min(len(RESOURCE_POOL),
                         (score - 1) + rng.choice([0, 0, 1, -1])))
    if n_kinds == 0:
        return rng.choice([
            "RESOURCES:\nNone listed.",
            "RESOURCES:\nTLMs will be provided.",   # names nothing on purpose
        ])
    kinds = rng.sample(sorted(RESOURCE_POOL), n_kinds)
    items = [rng.choice(RESOURCE_POOL[kind]).format(kw=keywords[i % len(keywords)])
             for i, kind in enumerate(kinds)]
    return "RESOURCES:\n" + "; ".join(items) + "."


def _differentiation_block(rng, score: int) -> str:
    n_strategies = max(0, min(len(DIFFERENTIATION_POOL),
                              (score - 1) + rng.choice([0, 0, 1, -1])))
    if n_strategies == 0:
        return rng.choice([
            "DIFFERENTIATION:\nNo reference to differing learner needs.",
            "DIFFERENTIATION:\nAll learners will do the same task.",
        ])
    kinds = rng.sample(sorted(DIFFERENTIATION_POOL), n_strategies)
    return "DIFFERENTIATION:\n" + " ".join(
        rng.choice(DIFFERENTIATION_POOL[kind]) for kind in kinds)


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


def _intro_block(rng, score: int, topic: str, keywords: list[str]) -> str:
    """Introduction, built from independently-sampled quality components.

    Returns "" for some weak plans so that "the section is missing entirely" is
    a state the model actually sees — which is what F20's first component and
    the missing-section suggestion rules key on.
    """
    if score == 1 and rng.random() < 0.45:
        return ""

    lines = []
    if _graded(rng, score):
        lines.append("Teacher opens with a short hook to settle the class.")
    if _graded(rng, score):
        # Prior-knowledge cue.
        lines.append(rng.choice([
            f"Teacher reviews learners' previous knowledge of {keywords[-1]}.",
            f"Learners recall what they already know about {keywords[-1]}.",
            f"Teacher revises the previous lesson on {keywords[-1]}.",
        ]))
    if _graded(rng, score):
        # The link that makes it a real RPK stage: shares vocabulary with the
        # objectives rather than gesturing at "previous knowledge" generically.
        lines.append(
            f"This is connected to today's topic, {topic}, and in particular to "
            f"{keywords[0]}.")
    if _graded(rng, score):
        lines.append("The objective is shared with learners at the start.")

    if not lines:
        lines.append("Teacher begins the lesson.")
    return "INTRODUCTION:\n" + " ".join(lines)


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


def _closure_block(rng, score: int) -> str:
    """Closure, built from the three things a closure is judged on.

    Consolidation, learner involvement and an attainment check are sampled
    independently, so a plan can summarise without involving learners — which is
    the common real weakness F22 is meant to detect.
    """
    if score == 1 and rng.random() < 0.45:
        return ""

    lines = []
    if _graded(rng, score):
        lines.append(rng.choice([
            "Teacher summarises the key points of the lesson.",
            "The main points are consolidated on the board.",
        ]))
    if _graded(rng, score):
        lines.append(rng.choice([
            "Learners summarise in their own words what they have learnt.",
            "Learners state the key points to the class.",
        ]))
    if _graded(rng, score):
        lines.append(rng.choice([
            "Closing questions check whether the objective was met.",
            "Teacher asks two questions to confirm attainment before dismissal.",
        ]))

    if not lines:
        lines.append("Lesson ends.")
    return "CLOSURE:\n" + " ".join(lines)


def _activities_block(rng, score: int, keywords: list[str], with_time: bool) -> str:
    # Jittered for the same reason as the assessment items: counts that map
    # one-to-one onto a score turn the derived feature into the label.
    n_learner = max(0, min(6, [0, 1, 2, 3, 4][score] + rng.choice([-1, 0, 0, 1])))
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
    # The item count must track quality WITHOUT being a function of it. It used
    # to be exactly `score`, which made assessment_item_count (F14) a verbatim
    # copy of the label: cross-validation reported QWK 1.000 with zero variance
    # for this criterion, a number that measured the leak and nothing else.
    # Overlapping ranges keep the feature informative and the task real.
    n_items = max(0, min(6, score + rng.choice([-1, 0, 0, 1])))
    # What actually separates a weak assessment from a strong one is whether the
    # items test what the objectives promised — the signal F13 measures.
    offtopic_chance = max(0.0, (3 - score) / 3.0)
    lines = ["EVALUATION:"]
    if n_items == 0:
        lines.append("Exercise to be given later.")
        return "\n".join(lines)
    for i in range(n_items):
        if rng.random() < offtopic_chance:
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

    # Written quality is tied to the criterion F17/F18 actually map to. Deriving
    # it from the mean of all ten scores leaked the entire quality profile into
    # every plan's readability features, which inflated held-out QWK for the
    # four criteria no structural feature measures.
    language_quality = scores["concept_explanation_examples"]

    parts = [
        header,
        _objectives_block(rng, scores["learning_outcomes"], kws, topic),
        _pedagogical_content_block(scores["pedagogical_content_knowledge"], kws),
        _resources_block(rng, scores["resources_including_ict"], kws),
        _intro_block(rng, scores["lesson_introduction_rpk"], topic, kws),
        _content_block(rng, scores["pedagogical_content_knowledge"], topic, kws),
        _activities_block(rng, scores["teaching_learning_strategies"], kws, with_time),
        _assessment_block(rng, scores["assessment_strategies_in_plan"], kws),
        _differentiation_block(rng, scores["attention_to_all_learners"]),
        _explanation_block(scores["concept_explanation_examples"], kws),
        _sequencing_block(scores["lesson_sequencing"], with_time),
        _closure_block(rng, scores["lesson_closure"]),
    ]
    # Weak plans can omit the introduction or closure entirely.
    parts = [part for part in parts if part.strip()]
    text = _apply_language_quality(rng, "\n\n".join(parts), language_quality)
    return SyntheticPlan(plan_id=plan_id, text=text, scores=scores,
                         subject=subject, topic=topic)


def generate_dataset(n: int = 200, seed: int = 42) -> list[SyntheticPlan]:
    rng = random.Random(seed)
    return [generate_plan(rng, f"synth_{i:04d}") for i in range(n)]


# --------------------------------------------------------------------------
# Revision pairs
# --------------------------------------------------------------------------

@dataclass
class RevisionPair:
    """The same lesson before and after a revision, with known score deltas.

    The project brief calls for synthetic lesson-plan *revisions*, not only
    independent plans, for three reasons:

      * the dashboard has a revision-history view, which needs versioned plans
        of the same lesson to have anything to show;
      * objective 24 measures lesson-plan quality *before and after* using the
        dashboard, and that pipeline needs test data with a known delta;
      * a model trained only on unrelated plans has never seen what an
        improvement to a given lesson looks like.

    Both versions are rendered from the same seed, so subject, topic and
    vocabulary carry over and only the quality-driven text changes — the way a
    real resubmission would.
    """

    pair_id: str
    before: SyntheticPlan
    after: SyntheticPlan
    improved: list[str]          # criteria raised in the revision

    @property
    def score_delta(self) -> dict[str, int]:
        return {d: self.after.scores[d] - self.before.scores[d]
                for d in RUBRIC_DIMENSIONS}


def generate_revision_pair(seed: int, pair_id: str,
                           n_improved: int = 3) -> RevisionPair:
    """One before/after pair for the same lesson."""
    chooser = random.Random(seed)
    base = chooser.choice([1, 2, 2, 3])          # revisions start from weaker plans
    before_scores = {
        d: min(4, max(1, base + chooser.choice([-1, 0, 0, 1])))
        for d in RUBRIC_DIMENSIONS
    }

    improvable = [d for d in RUBRIC_DIMENSIONS if before_scores[d] < 4]
    improved = sorted(chooser.sample(improvable, min(n_improved, len(improvable))))
    after_scores = dict(before_scores)
    for dimension in improved:
        after_scores[dimension] = min(4, before_scores[dimension]
                                      + chooser.choice([1, 1, 2]))

    # Fresh generators from the same seed: identical subject/topic/keywords,
    # divergence only where the quality profile actually differs.
    before = generate_plan(random.Random(seed), f"{pair_id}_v1", before_scores)
    after = generate_plan(random.Random(seed), f"{pair_id}_v2", after_scores)
    return RevisionPair(pair_id=pair_id, before=before, after=after,
                        improved=improved)


def generate_revisions(n: int = 40, seed: int = 42) -> list[RevisionPair]:
    rng = random.Random(seed)
    return [generate_revision_pair(rng.randrange(2 ** 31), f"rev_{i:04d}")
            for i in range(n)]


def write_revisions(pairs: list[RevisionPair], out_dir: str | Path) -> Path:
    """Write both versions of each pair plus a revisions.csv manifest.

    Plans go into the same ``plans/`` directory and the same labels.csv shape as
    ``write_dataset``, so revision plans are usable as ordinary training rows.
    ``revisions.csv`` carries the pairing S3's revision history and S5's
    before/after analysis need.
    """
    out_dir = Path(out_dir)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)

    flat: list[SyntheticPlan] = []
    for pair in pairs:
        flat.extend((pair.before, pair.after))
    write_dataset(flat, out_dir)

    manifest = out_dir / "revisions.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "before_plan_id", "after_plan_id",
                         "improved_criteria", "overall_before", "overall_after",
                         "overall_delta"])
        for pair in pairs:
            before_overall = overall_score(pair.before.scores)
            after_overall = overall_score(pair.after.scores)
            writer.writerow([
                pair.pair_id, pair.before.plan_id, pair.after.plan_id,
                "|".join(pair.improved),
                round(before_overall, 2), round(after_overall, 2),
                round(after_overall - before_overall, 2),
            ])
    return manifest


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
    parser.add_argument("--revisions", type=int, default=0,
                        help="also generate N before/after revision pairs "
                             "(written as 2N extra plans + revisions.csv)")
    args = parser.parse_args()

    if args.revisions:
        pairs = generate_revisions(args.revisions, args.seed)
        revision_manifest = write_revisions(pairs, args.out)
        gained = sum(1 for p in pairs
                     if overall_score(p.after.scores) > overall_score(p.before.scores))
        print(f"Wrote {len(pairs)} revision pairs ({2 * len(pairs)} plans), "
              f"{gained} with a positive delta -> {revision_manifest}")
    else:
        dataset = generate_dataset(args.n, args.seed)
        manifest_path = write_dataset(dataset, args.out)
        print(f"Wrote {len(dataset)} plans -> {manifest_path}")
