# app/study/export.py
"""Analysis-ready CSVs for S5. Participant codes only.

Three files, joinable on ``participant_code``:

    participants.csv   one row per consenting participant
    responses.csv      one row per survey (pre and post share the same columns)
    submissions.csv    one row per scored plan, tagged with where it fell
                       relative to the two surveys

What is left out is as deliberate as what is in. No email, name or plan file
name ever reaches a CSV — file names are often the student's own name.
Declined and withdrawn people are absent entirely, and so is anyone whose role
is not a participant role.

**Pilot data.** Unless ``include_pilot`` is set, only material collected under
the approved instruments is exported: responses carrying a protocol number,
from people who consented to the approved documents. Before approval that is
nobody, and the CLI says so rather than writing three empty files silently.

Item columns hold the raw response. Scale columns (``TR_mean`` etc.) are
reverse-keyed, so TR6 enters ``TR_mean`` as 6 − response; see
``docs/irb/survey_instruments.md``. No statistics are computed here: that is
S5's layer.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from app.database import db
from app.study import approval, flow, instruments

PARTICIPANT_COLUMNS = [
    "participant_code", "consented_at", "pilot_consent", "pre_completed",
    "post_completed", "pre_after_feedback", "plans_between_surveys",
    "total_plans",
]

RESPONSE_COLUMNS = (
    ["participant_code", "wave", "submitted_at", "duration_seconds",
     "submissions_before", "pre_after_feedback", "instrument_version",
     "irb_protocol", "pilot"]
    + [item.id for item in instruments.ITEMS]
    + [b.id for b in instruments.BACKGROUND]
    + [f"{s.key}_mean" for s in instruments.SCALES]
)

SUBMISSION_BASE_COLUMNS = [
    "participant_code", "sequence", "submitted_at", "phase", "overall_score",
    "band", "contract_version",
]

Table = tuple[list[str], list[dict[str, Any]]]


def _iso(value) -> str:
    return value.isoformat(sep=" ", timespec="seconds") if value else ""


def _blank(value) -> Any:
    return "" if value is None else value


def _phase(sequence: int, waves: dict[str, int]) -> str:
    if "pre" not in waves:
        return "no_pre_survey"
    if sequence <= waves["pre"]:
        return "before_pre"
    if "post" not in waves or sequence <= waves["post"]:
        return "between_surveys"
    return "after_post"


def build(include_pilot: bool = False) -> dict[str, Table]:
    """``{name: (columns, rows)}`` for the three files."""
    tables: dict[str, Table] = {
        "participants": (PARTICIPANT_COLUMNS, []),
        "responses": (RESPONSE_COLUMNS, []),
        "submissions": (SUBMISSION_BASE_COLUMNS, []),
    }
    engine = db.get_engine()
    if engine is None:
        return tables

    consents, responses, submissions = (
        db.consents, db.survey_responses, db.submissions)
    with engine.connect() as connection:
        people = connection.execute(
            select(consents.c.user_email, consents.c.participant_code,
                   consents.c.decided_at, consents.c.materials_fingerprint)
            .join(db.users, db.users.c.email == consents.c.user_email)
            .where(consents.c.decision == "agreed",
                   consents.c.withdrawn_at.is_(None),
                   db.users.c.role.in_(sorted(flow.PARTICIPANT_ROLES)))
            .order_by(consents.c.participant_code)).mappings().all()

        approved = approval.APPROVED_FINGERPRINT if approval.IRB_PROTOCOL else None
        if not include_pilot:
            people = [p for p in people if p["materials_fingerprint"] == approved]
        by_email = {p["user_email"]: p for p in people}
        if not by_email:
            return tables

        survey_rows = connection.execute(
            select(responses)
            .where(responses.c.user_email.in_(list(by_email)))
            .order_by(responses.c.user_email, responses.c.id)).mappings().all()
        plan_rows = connection.execute(
            select(submissions.c.user_email, submissions.c.submitted_at,
                   submissions.c.overall_score, submissions.c.band,
                   submissions.c.rubric_scores, submissions.c.contract_version)
            .where(submissions.c.user_email.in_(list(by_email)))
            .order_by(submissions.c.user_email, submissions.c.submitted_at,
                      submissions.c.id)).mappings().all()

    # Phases come from every response, pilot or not: when a survey was taken
    # is a fact about the timeline even if its answers are not exported.
    waves: dict[str, dict[str, int]] = {}
    for row in survey_rows:
        waves.setdefault(row["user_email"], {})[row["wave"]] = row["submissions_before"]

    plans: dict[str, int] = {}
    criteria: set[str] = set()
    submission_out: list[dict[str, Any]] = []
    for row in plan_rows:
        email = row["user_email"]
        plans[email] = plans.get(email, 0) + 1
        scores = row["rubric_scores"] or {}
        criteria.update(scores)
        submission_out.append({
            "participant_code": by_email[email]["participant_code"],
            "sequence": plans[email],
            "submitted_at": _iso(row["submitted_at"]),
            "phase": _phase(plans[email], waves.get(email, {})),
            "overall_score": ("" if row["overall_score"] is None
                              else float(row["overall_score"])),
            "band": _blank(row["band"]),
            "contract_version": _blank(row["contract_version"]),
            **scores,
        })
    tables["submissions"] = (SUBMISSION_BASE_COLUMNS + sorted(criteria),
                             submission_out)

    response_out: list[dict[str, Any]] = []
    for row in survey_rows:
        if row["irb_protocol"] is None and not include_pilot:
            continue
        answers = row["answers"] or {}
        out = {
            "participant_code": by_email[row["user_email"]]["participant_code"],
            "wave": row["wave"],
            "submitted_at": _iso(row["submitted_at"]),
            "duration_seconds": _blank(row["duration_seconds"]),
            "submissions_before": row["submissions_before"],
            "pre_after_feedback": (int(row["submissions_before"] > 0)
                                   if row["wave"] == "pre" else ""),
            "instrument_version": row["instrument_version"],
            "irb_protocol": _blank(row["irb_protocol"]),
            "pilot": int(row["irb_protocol"] is None),
        }
        for column in RESPONSE_COLUMNS:
            if column in answers:
                out[column] = answers[column]
        for key, mean in instruments.scale_means(answers, row["wave"]).items():
            out[f"{key}_mean"] = _blank(mean)
        response_out.append(out)
    tables["responses"] = (RESPONSE_COLUMNS, response_out)

    participant_out: list[dict[str, Any]] = []
    for email, person in by_email.items():
        mine = waves.get(email, {})
        participant_out.append({
            "participant_code": person["participant_code"],
            "consented_at": _iso(person["decided_at"]),
            "pilot_consent": int(person["materials_fingerprint"] != approved),
            "pre_completed": int("pre" in mine),
            "post_completed": int("post" in mine),
            "pre_after_feedback": (int(mine["pre"] > 0) if "pre" in mine else ""),
            "plans_between_surveys": (mine["post"] - mine["pre"]
                                      if {"pre", "post"} <= mine.keys() else ""),
            "total_plans": plans.get(email, 0),
        })
    tables["participants"] = (PARTICIPANT_COLUMNS, participant_out)
    return tables


def to_csv(table: Table) -> str:
    columns, rows = table
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, restval="",
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write(out_dir: Path, include_pilot: bool = False,
          tables: Optional[dict[str, Table]] = None) -> list[Path]:
    """Write the three CSVs. UTF-8 with a BOM, so Excel opens them correctly."""
    tables = build(include_pilot) if tables is None else tables
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        path.write_text(to_csv(table), encoding="utf-8-sig", newline="")
        written.append(path)
    return written
