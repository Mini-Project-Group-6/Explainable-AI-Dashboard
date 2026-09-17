# analysis/synthetic.py
"""A synthetic export with known effects, to exercise the analysis.

Written in exactly the export's format (``app.study.export`` column lists), so
the analysis cannot tell it from a real export except for the ``SYNTHETIC``
marker file beside the CSVs, which puts a banner on every report built from it.

What is built in, so a test can check the analysis finds it:

* every paired scale rises between the surveys (``SHIFTS``), trust the most;
* items of one scale share a person-level latent value, so alpha is high —
  and TR6 is written *reverse-worded*, so alpha only comes out high if the
  analysis reverse-keys it;
* plan scores rise with each draft, faster for people whose trust rose more;
* some participants skip the follow-up, and some answer the first survey only
  after seeing feedback, so both exclusions have something to exclude.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.load import SYNTHETIC_MARKER
from app.study import export, instruments

SHIFTS = {"PU": 0.30, "PEOU": 0.20, "BI": 0.25, "TR": 0.45}
BASELINES = {"PU": 3.3, "PEOU": 3.6, "BI": 3.4, "TR": 3.1, "ES": 3.5}

#: The ten rubric criterion keys, as the dashboard stores them. Listed here
#: rather than imported so the analysis does not need the model stack.
CRITERIA = (
    "learning_outcomes", "pedagogical_content_knowledge",
    "teaching_learning_strategies", "resources_including_ict",
    "assessment_strategies_in_plan", "lesson_introduction_rpk",
    "lesson_sequencing", "attention_to_all_learners",
    "concept_explanation_examples", "lesson_closure",
)


def _answers(rng, wave: str, level: dict[str, float]) -> dict:
    answers: dict = {}
    for item in instruments.items_for(wave):
        keyed = int(np.clip(np.rint(level[item.scale] + rng.normal(0, 0.55)), 1, 5))
        # Stored raw, as a participant would have answered: a reverse-worded
        # item's raw answer is the mirror of its keyed value.
        answers[item.id] = (6 - keyed) if item.reverse else keyed
    for question in instruments.background_for(wave):
        if rng.random() < 0.9:
            answers[question.id] = str(rng.choice(question.options[:-1]))
    return answers


def write_synthetic_export(out_dir: Path, n: int = 60, seed: int = 7,
                           attrition: float = 0.1, late: float = 0.05) -> Path:
    rng = np.random.default_rng(seed)
    participants, responses, submissions = [], [], []

    for index in range(n):
        code = f"P-SYN{index:05d}"
        latent = {key: rng.normal(0, 0.6) for key in BASELINES}
        trust_gain = SHIFTS["TR"] + rng.normal(0, 0.5)
        pre_level = {key: BASELINES[key] + latent[key] for key in BASELINES}
        post_level = {key: pre_level[key] + SHIFTS.get(key, 0) + rng.normal(0, 0.3)
                      for key in BASELINES}
        post_level["TR"] = pre_level["TR"] + trust_gain

        late_baseline = rng.random() < late
        has_post = rng.random() >= attrition
        before_pre = 1 if late_baseline else 0
        plans = max(int(rng.integers(2, 6)), before_pre + 2)
        waves = {"pre": before_pre}
        if has_post:
            waves["post"] = before_pre + 2

        for wave, level in (("pre", pre_level), ("post", post_level)):
            if wave not in waves:
                continue
            answers = _answers(rng, wave, level)
            row = {
                "participant_code": code, "wave": wave,
                "submitted_at": f"2026-10-{1 + index % 28:02d} 10:00:00",
                "duration_seconds": int(rng.integers(240, 900)),
                "submissions_before": waves[wave],
                "pre_after_feedback": int(late_baseline) if wave == "pre" else "",
                "instrument_version": instruments.INSTRUMENT_VERSION,
                "irb_protocol": "SYNTHETIC", "pilot": 0, **answers,
            }
            for key, mean in instruments.scale_means(answers, wave).items():
                row[f"{key}_mean"] = "" if mean is None else mean
            responses.append(row)

        score = float(np.clip(rng.normal(42, 14), 0, 100))
        for sequence in range(1, plans + 1):
            if sequence > 1:
                score = float(np.clip(score + 4 + 6 * trust_gain + rng.normal(0, 4),
                                      0, 100))
            criteria = {key: int(np.clip(np.rint(1 + 3 * score / 100
                                                 + rng.normal(0, 0.4)), 1, 4))
                        for key in CRITERIA}
            submissions.append({
                "participant_code": code, "sequence": sequence,
                "submitted_at": f"2026-10-{1 + index % 28:02d} {10 + sequence}:00:00",
                "phase": export._phase(sequence, waves),
                "overall_score": round(score, 2), "band": "",
                "contract_version": "2.0.0", **criteria,
            })

        participants.append({
            "participant_code": code, "consented_at": "2026-10-01 09:00:00",
            "pilot_consent": 0, "pre_completed": 1,
            "post_completed": int(has_post),
            "pre_after_feedback": int(late_baseline),
            "plans_between_surveys": 2 if has_post else "",
            "total_plans": plans,
        })

    out_dir = Path(out_dir)
    export.write(out_dir, tables={
        "participants": (export.PARTICIPANT_COLUMNS, participants),
        "responses": (export.RESPONSE_COLUMNS, responses),
        "submissions": (export.SUBMISSION_BASE_COLUMNS + list(CRITERIA), submissions),
    })
    (out_dir / SYNTHETIC_MARKER).write_text(
        f"Generated by python -m analysis synthetic (n={n}, seed={seed}).\n"
        "These are not real participants. Reports built from this folder say so.\n",
        encoding="utf-8")
    return out_dir
