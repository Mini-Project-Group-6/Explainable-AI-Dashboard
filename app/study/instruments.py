# app/study/instruments.py
"""The survey instruments. S4 owns this file.

Published scales adapted to lesson planning, all on one 5-point agreement
scale:

    PU    Perceived usefulness       Davis (1989)              6 items  pre + post
    PEOU  Perceived ease of use      Davis (1989)              6 items  pre + post
    BI    Behavioural intention      Venkatesh & Davis (2000)  2 items  pre + post
    TR    Trust in the tool          Hoffman et al. (2018)     8 items  pre + post
    ES    Explanation satisfaction   Hoffman et al. (2018)     8 items  post only

TAM and trust are asked word for word the same in both waves, so each pre/post
pair is a like-for-like comparison. Explanation satisfaction is post-only:
before a participant has seen an explanation there is nothing to be satisfied
with.

Every item keeps the published wording it was adapted from (``source_text``)
beside the wording participants see (``text``). The protocol appendix and the
methods section both need that adaptation table; generating it from here
(``python -m app.study instruments --write``) means it cannot drift from what
the dashboard actually asks.

Section headings shown to participants are deliberately not the construct
names. "Perceived usefulness" as a heading tells the respondent what the items
are meant to measure, which is a known source of inflated inter-item
correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

INSTRUMENT_VERSION = "0.1.0"

WAVES = ("pre", "post")

LIKERT = (1, 2, 3, 4, 5)
LIKERT_LABELS = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neither agree nor disagree",
    4: "Agree",
    5: "Strongly agree",
}

SOURCES = {
    "davis1989": (
        "Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and "
        "user acceptance of information technology. MIS Quarterly, 13(3), "
        "319–340."),
    "venkatesh2000": (
        "Venkatesh, V., & Davis, F. D. (2000). A theoretical extension of the "
        "technology acceptance model: Four longitudinal field studies. "
        "Management Science, 46(2), 186–204."),
    "hoffman2018": (
        "Hoffman, R. R., Mueller, S. T., Klein, G., & Litman, J. (2018). "
        "Metrics for explainable AI: Challenges and prospects. "
        "arXiv:1812.04608."),
}

#: Sources whose ``source_text`` has been compared word for word against the
#: published paper. The wording below was transcribed for this draft, not
#: copied from the PDFs, so none is verified yet. Add a key once someone has
#: checked it; ``approval.approval_problems()`` reports the rest.
_VERIFIED_SOURCES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Scale:
    key: str
    name: str
    heading: str
    source: str
    waves: tuple[str, ...] = WAVES


@dataclass(frozen=True)
class Item:
    id: str
    scale: str
    text: str
    source_text: str
    reverse: bool = False


@dataclass(frozen=True)
class BackgroundItem:
    """A categorical question. Optional, and asked in the pre-survey only."""
    id: str
    text: str
    options: tuple[str, ...]


_USING = "Using the dashboard"

SCALES = (
    Scale("PU", "Perceived usefulness", _USING, "davis1989"),
    Scale("PEOU", "Perceived ease of use", _USING, "davis1989"),
    Scale("BI", "Behavioural intention", _USING, "venkatesh2000"),
    Scale("TR", "Trust", "Relying on the dashboard", "hoffman2018"),
    Scale("ES", "Explanation satisfaction", "The explanations of your scores",
          "hoffman2018", waves=("post",)),
)

ITEMS = (
    # Davis (1989), perceived usefulness. Original system: CHART-MASTER.
    Item("PU1", "PU",
         "Using the dashboard would enable me to prepare lesson plans more "
         "quickly.",
         "Using CHART-MASTER in my job would enable me to accomplish tasks "
         "more quickly."),
    Item("PU2", "PU",
         "Using the dashboard would improve my lesson planning.",
         "Using CHART-MASTER would improve my job performance."),
    Item("PU3", "PU",
         "Using the dashboard would increase my productivity in lesson "
         "planning.",
         "Using CHART-MASTER in my job would increase my productivity."),
    Item("PU4", "PU",
         "Using the dashboard would enhance my effectiveness in lesson "
         "planning.",
         "Using CHART-MASTER would enhance my effectiveness on the job."),
    Item("PU5", "PU",
         "Using the dashboard would make it easier to plan lessons.",
         "Using CHART-MASTER would make it easier to do my job."),
    Item("PU6", "PU",
         "I would find the dashboard useful in my lesson planning.",
         "I would find CHART-MASTER useful in my job."),

    # Davis (1989), perceived ease of use.
    Item("PEOU1", "PEOU",
         "Learning to use the dashboard would be easy for me.",
         "Learning to operate CHART-MASTER would be easy for me."),
    Item("PEOU2", "PEOU",
         "I would find it easy to get the dashboard to do what I want it to "
         "do.",
         "I would find it easy to get CHART-MASTER to do what I want it to "
         "do."),
    Item("PEOU3", "PEOU",
         "My interaction with the dashboard would be clear and "
         "understandable.",
         "My interaction with CHART-MASTER would be clear and "
         "understandable."),
    Item("PEOU4", "PEOU",
         "I would find the dashboard to be flexible to interact with.",
         "I would find CHART-MASTER to be flexible to interact with."),
    Item("PEOU5", "PEOU",
         "It would be easy for me to become skilful at using the dashboard.",
         "It would be easy for me to become skillful at using CHART-MASTER."),
    Item("PEOU6", "PEOU",
         "I would find the dashboard easy to use.",
         "I would find CHART-MASTER easy to use."),

    # Venkatesh & Davis (2000), behavioural intention.
    Item("BI1", "BI",
         "Assuming I have access to the dashboard, I intend to use it for my "
         "lesson plans.",
         "Assuming I have access to the system, I intend to use it."),
    Item("BI2", "BI",
         "Given that I have access to the dashboard, I predict that I would "
         "use it.",
         "Given that I have access to the system, I predict that I would use "
         "it."),

    # Hoffman et al. (2018), trust scale recommended for XAI.
    Item("TR1", "TR",
         "I am confident in the dashboard. I feel that it works well.",
         "I am confident in the [tool]. I feel that it works well."),
    Item("TR2", "TR",
         "The outputs of the dashboard are very predictable.",
         "The outputs of the [tool] are very predictable."),
    Item("TR3", "TR",
         "The dashboard is very reliable. I can count on it to be correct all "
         "the time.",
         "The tool is very reliable. I can count on it to be correct all the "
         "time."),
    Item("TR4", "TR",
         "I feel safe that when I rely on the dashboard I will get the right "
         "answers.",
         "I feel safe that when I rely on the [tool] I will get the right "
         "answers."),
    Item("TR5", "TR",
         "The dashboard is efficient in that it works very quickly.",
         "The [tool] is efficient in that it works very quickly."),
    Item("TR6", "TR",
         "I am wary of the dashboard.",
         "I am wary of the [tool].",
         reverse=True),
    Item("TR7", "TR",
         "The dashboard can assess a lesson plan better than a novice human "
         "assessor.",
         "The [tool] can perform the task better than a novice human user."),
    Item("TR8", "TR",
         "I like using the dashboard to decide how to improve my lesson "
         "plans.",
         "I like using the system for decision making."),

    # Hoffman et al. (2018), explanation satisfaction scale.
    Item("ES1", "ES",
         "From the explanations, I understand how the dashboard scores a "
         "lesson plan.",
         "From the explanation, I understand how the [tool] works."),
    Item("ES2", "ES",
         "The explanations of how the dashboard scores a lesson plan are "
         "satisfying.",
         "This explanation of how the [tool] works is satisfying."),
    Item("ES3", "ES",
         "The explanations of how the dashboard scores a lesson plan have "
         "sufficient detail.",
         "This explanation of how the [tool] works has sufficient detail."),
    Item("ES4", "ES",
         "The explanations of how the dashboard scores a lesson plan seem "
         "complete.",
         "This explanation of how the [tool] works seems complete."),
    Item("ES5", "ES",
         "The explanations tell me how to use the dashboard.",
         "This explanation of how the [tool] works tells me how to use it."),
    Item("ES6", "ES",
         "The explanations are useful to my goals.",
         "This explanation of how the [tool] works is useful to my goals."),
    Item("ES7", "ES",
         "The explanations show me how accurate the dashboard is.",
         "This explanation of the [tool] shows me how accurate the [tool] "
         "is."),
    Item("ES8", "ES",
         "The explanations let me judge when I should trust and not trust "
         "the dashboard.",
         "This explanation lets me judge when I should trust and not trust "
         "the [tool]."),
)

#: Prior AI use is the covariate most likely to moderate trust, and year of
#: study the one most likely to moderate plan quality. Nothing else is asked:
#: every extra demographic is data the protocol has to justify and protect.
BACKGROUND = (
    BackgroundItem("BG1", "Which level are you in?",
                   ("Level 100", "Level 200", "Level 300", "Level 400",
                    "Prefer not to say")),
    BackgroundItem("BG2",
                   "Before this study, how often had you used AI tools such as "
                   "ChatGPT?",
                   ("Never", "Rarely", "Sometimes", "Often",
                    "Prefer not to say")),
)

_SCALES_BY_KEY = {scale.key: scale for scale in SCALES}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def scale(key: str) -> Scale:
    return _SCALES_BY_KEY[key]


def scales_for(wave: str) -> tuple[Scale, ...]:
    _check_wave(wave)
    return tuple(s for s in SCALES if wave in s.waves)


def items_for(wave: str) -> tuple[Item, ...]:
    """The Likert items asked in *wave*, in the order they are shown."""
    keys = {s.key for s in scales_for(wave)}
    return tuple(item for item in ITEMS if item.scale in keys)


def background_for(wave: str) -> tuple[BackgroundItem, ...]:
    _check_wave(wave)
    return BACKGROUND if wave == "pre" else ()


def sections_for(wave: str) -> list[tuple[str, list[Item]]]:
    """Items grouped under their participant-facing headings, in order."""
    sections: list[tuple[str, list[Item]]] = []
    for item in items_for(wave):
        heading = scale(item.scale).heading
        if not sections or sections[-1][0] != heading:
            sections.append((heading, []))
        sections[-1][1].append(item)
    return sections


def unverified_sources() -> list[str]:
    used = {s.source for s in SCALES}
    return sorted(used - _VERIFIED_SOURCES)


def _check_wave(wave: str) -> None:
    if wave not in WAVES:
        raise ValueError(f"Unknown survey wave {wave!r}; expected one of {WAVES}.")


# ---------------------------------------------------------------------------
# Validation and scoring
# ---------------------------------------------------------------------------

def _join_numbers(numbers: list[int]) -> str:
    text = [str(n) for n in numbers]
    return text[0] if len(text) == 1 else ", ".join(text[:-1]) + " and " + text[-1]


def problems(answers: Mapping[str, Any], wave: str) -> list[str]:
    """Why *answers* cannot be stored as a *wave* response. Empty when valid.

    Every Likert item is required: a scale mean over a subset of its items is
    not the published scale. Background questions are optional.
    """
    items = items_for(wave)
    background = background_for(wave)
    known = {item.id for item in items} | {b.id for b in background}
    found: list[str] = []

    unknown = sorted(set(answers) - known)
    if unknown:
        found.append(f"Unexpected answers for {', '.join(unknown)}.")

    missing = [n for n, item in enumerate(items, 1)
               if answers.get(item.id) is None]
    if missing:
        noun = "Statement" if len(missing) == 1 else "Statements"
        verb = "has" if len(missing) == 1 else "have"
        found.append(f"{noun} {_join_numbers(missing)} {verb} no answer.")

    for item in items:
        value = answers.get(item.id)
        if value is not None and (isinstance(value, bool) or value not in LIKERT):
            found.append(f"{item.id} must be one of {LIKERT}, not {value!r}.")

    for question in background:
        value = answers.get(question.id)
        if value is not None and value not in question.options:
            found.append(f"{question.id} has an unrecognised answer {value!r}.")
    return found


def keyed(item: Item, value: int) -> int:
    """The value with reverse-worded items flipped, so higher always means more."""
    return (LIKERT[0] + LIKERT[-1] - value) if item.reverse else value


def scale_means(answers: Mapping[str, Any], wave: str) -> dict[str, Optional[float]]:
    """Mean of each scale's items after reverse keying. ``None`` if incomplete."""
    means: dict[str, Optional[float]] = {}
    for s in scales_for(wave):
        members = [item for item in ITEMS if item.scale == s.key]
        values = [answers.get(item.id) for item in members]
        if any(v is None for v in values):
            means[s.key] = None
            continue
        means[s.key] = round(
            sum(keyed(item, v) for item, v in zip(members, values)) / len(values), 4)
    return means


# ---------------------------------------------------------------------------
# Adaptation table
# ---------------------------------------------------------------------------

def canonical() -> dict[str, Any]:
    """Everything a participant is asked, in a stable shape for fingerprinting."""
    return {
        "version": INSTRUMENT_VERSION,
        "likert": [[v, LIKERT_LABELS[v]] for v in LIKERT],
        "scales": [[s.key, s.heading, list(s.waves)] for s in SCALES],
        "items": [[i.id, i.scale, i.text, i.reverse] for i in ITEMS],
        "background": [[b.id, b.text, list(b.options)] for b in BACKGROUND],
    }


def adaptation_markdown() -> str:
    """The instrument appendix for the protocol, generated from this module."""
    lines = [
        "# Survey instruments",
        "",
        "<!-- Generated by `python -m app.study instruments --write`. "
        "Do not edit by hand: change app/study/instruments.py and regenerate. -->",
        "",
        f"Instrument version **{INSTRUMENT_VERSION}**. Every statement is "
        "answered on the same five-point scale:",
        "",
        " · ".join(f"{v} = {LIKERT_LABELS[v]}" for v in LIKERT),
        "",
        "The pre-survey is answered before a participant's first lesson plan "
        "is scored; the post-survey opens once they have had further plans "
        "scored. The usefulness, ease-of-use, intention and trust statements "
        "are identical in both surveys. Explanation satisfaction is asked in "
        "the post-survey only.",
        "",
    ]
    unverified = unverified_sources()
    if unverified:
        lines += [
            "> **Draft.** The *published wording* column has not yet been "
            "checked word for word against these sources: "
            f"{', '.join(unverified)}.",
            "",
        ]
    lines += [
        "## Scales",
        "",
        "| Scale | Construct | Items | Surveys | Source |",
        "|---|---|---|---|---|",
    ]
    for s in SCALES:
        count = sum(1 for item in ITEMS if item.scale == s.key)
        lines.append(f"| {s.key} | {s.name} | {count} | "
                     f"{', '.join(s.waves)} | {SOURCES[s.source]} |")

    for s in SCALES:
        lines += [
            "",
            f"## {s.key} — {s.name}",
            "",
            f"Shown to participants under the heading *{s.heading}*.",
            "",
            "| Item | Wording used | Published wording | Reverse-scored |",
            "|---|---|---|---|",
        ]
        for item in ITEMS:
            if item.scale == s.key:
                lines.append(f"| {item.id} | {item.text} | {item.source_text} | "
                             f"{'yes' if item.reverse else ''} |")

    lines += [
        "",
        "## Background questions (pre-survey, optional)",
        "",
        "| Item | Question | Options |",
        "|---|---|---|",
    ]
    for b in BACKGROUND:
        lines.append(f"| {b.id} | {b.text} | {'; '.join(b.options)} |")

    lines += [
        "",
        "## Scoring",
        "",
        "A scale score is the mean of its items. Reverse-scored items are "
        f"keyed as {LIKERT[0] + LIKERT[-1]} − response first, so a higher "
        "score always means more of the construct. A scale is left blank "
        "unless every one of its items was answered; the dashboard does not "
        "accept a survey with a statement unanswered.",
        "",
        "## Adaptation notes",
        "",
        "- Davis (1989) worded items for a named system (CHART-MASTER) and a "
        "job. Here the system is *the dashboard* and the job is lesson "
        "planning. The conditional (\"would\") wording is kept in both "
        "surveys so the pre/post pairs are identical.",
        "- Hoffman et al. (2018) leave the system as a bracketed slot "
        "(\"[tool]\"). TR7's task and comparison group, and TR8's decision, "
        "are made specific to lesson-plan assessment and revision.",
        "- Spelling follows British English (\"skilful\"), as used in Ghanaian "
        "Colleges of Education.",
        "",
    ]
    return "\n".join(lines)
