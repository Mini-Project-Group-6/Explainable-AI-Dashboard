# app/study/store.py
"""Consent and survey reads and writes. The tables live in ``app.database.db``.

Nothing here is cached. Every read is a primary-key or indexed lookup, and a
stale consent state is the one thing this module must never serve: showing the
pre-survey to someone who has just withdrawn would be a protocol breach, not a
display glitch.
"""

from __future__ import annotations

import secrets
from typing import Any, Mapping, Optional

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database import db
from app.study import approval, flow, instruments

_CODE_ATTEMPTS = 8


def _new_code() -> str:
    """``P-`` plus 8 hex digits. Random, so it carries nothing about the person."""
    return f"P-{secrets.token_hex(4).upper()}"


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

def consent_record(email: str) -> Optional[flow.ConsentRecord]:
    engine = db.get_engine()
    if engine is None:
        return None
    with engine.connect() as connection:
        row = connection.execute(
            select(db.consents.c.decision, db.consents.c.participant_code,
                   db.consents.c.withdrawn_at)
            .where(db.consents.c.user_email == email)).mappings().first()
    if row is None:
        return None
    return flow.ConsentRecord(decision=row["decision"],
                              participant_code=row["participant_code"],
                              withdrawn=row["withdrawn_at"] is not None)


def record_consent(email: str, agreed: bool) -> Optional[flow.ConsentRecord]:
    """Record a decision and return the resulting record.

    Allowed moves: none → agreed/declined, and declined → agreed. An agreement
    is not turned back into a decline (that is what withdrawal is for), and a
    withdrawal is final here — rejoining goes through the research team.
    """
    engine = db.get_engine()
    if engine is None:
        return None

    try:
        existing = consent_record(email)
    except SQLAlchemyError:
        return None
    if existing is not None and (existing.withdrawn
                                 or existing.decision == "agreed"
                                 or not agreed):
        return existing

    decision = "agreed" if agreed else "declined"
    fingerprint = approval.fingerprint()
    for _ in range(_CODE_ATTEMPTS):
        code = _new_code() if agreed else None
        values = dict(decision=decision, participant_code=code,
                      materials_fingerprint=fingerprint)
        try:
            with engine.begin() as connection:
                if existing is None:
                    connection.execute(insert(db.consents).values(
                        user_email=email, **values))
                else:
                    connection.execute(
                        update(db.consents)
                        .where(db.consents.c.user_email == email)
                        .values(decided_at=func.now(), **values))
        except IntegrityError:
            # Either the code collided (retry with a new one) or a second
            # click already inserted this person's row (read it back).
            current = consent_record(email)
            if current is not None and current != existing:
                return current
            continue
        except SQLAlchemyError:
            return None
        return consent_record(email)
    raise RuntimeError("Could not allocate a unique participant code.")


def withdraw(email: str) -> bool:
    """Withdraw a participant and delete their survey answers.

    The consent row stays, marked withdrawn, so they are not asked again. Their
    scored plans stay in the dashboard's own history — that is the tool
    working, not study data — and the export leaves them out.
    """
    engine = db.get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as connection:
            changed = connection.execute(
                update(db.consents)
                .where(db.consents.c.user_email == email,
                       db.consents.c.decision == "agreed",
                       db.consents.c.withdrawn_at.is_(None))
                .values(withdrawn_at=func.now())).rowcount
            if changed:
                connection.execute(
                    delete(db.survey_responses)
                    .where(db.survey_responses.c.user_email == email))
    except SQLAlchemyError:
        # One transaction, so a failure leaves neither half applied.
        return False
    return bool(changed)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def completed_waves(email: str) -> dict[str, int]:
    """``{wave: plans scored when it was answered}`` for this person."""
    engine = db.get_engine()
    if engine is None:
        return {}
    with engine.connect() as connection:
        rows = connection.execute(
            select(db.survey_responses.c.wave,
                   db.survey_responses.c.submissions_before)
            .where(db.survey_responses.c.user_email == email)).all()
    return {wave: count for wave, count in rows}


def current_state(user) -> flow.StudyState:
    """The study state for a signed-in ``session.User``.

    A database that fails mid-session reads as unavailable: nothing asked,
    nothing blocked, rather than a traceback above the tabs.
    """
    if user.role not in flow.PARTICIPANT_ROLES:
        return flow.StudyState(flow.NOT_PARTICIPANT)
    if db.get_engine() is None:
        return flow.StudyState(flow.UNAVAILABLE)
    try:
        return flow.resolve(
            role=user.role,
            storage_ok=True,
            consent=consent_record(user.email),
            completed=completed_waves(user.email),
            submissions=db.submission_count(user.email),
        )
    except SQLAlchemyError:
        return flow.StudyState(flow.UNAVAILABLE)


def record_response(user, wave: str, answers: Mapping[str, Any],
                    duration_seconds: Optional[float] = None
                    ) -> tuple[bool, list[str]]:
    """Store one survey. Returns ``(saved, problems)``.

    Checked here rather than trusted from the page: the wave must be the one
    this person is due, and every answer must be valid for it.
    """
    state = current_state(user)
    if state.due_wave != wave:
        return False, [f"The {wave}-survey is not open for this account."]

    found = instruments.problems(answers, wave)
    if found:
        return False, found

    likert = {item.id for item in instruments.items_for(wave)}
    background = {b.id for b in instruments.background_for(wave)}
    clean: dict[str, Any] = {}
    for key, value in answers.items():
        if value is None:
            continue
        if key in likert:
            clean[key] = int(value)
        elif key in background:
            clean[key] = value

    engine = db.get_engine()
    try:
        with engine.begin() as connection:
            connection.execute(insert(db.survey_responses).values(
                user_email=user.email,
                wave=wave,
                answers=clean,
                submissions_before=db.submission_count(user.email),
                duration_seconds=(None if duration_seconds is None
                                  else int(round(duration_seconds))),
                instrument_version=instruments.INSTRUMENT_VERSION,
                materials_fingerprint=approval.fingerprint(),
                irb_protocol=approval.protocol_for_record(),
            ))
    except IntegrityError:
        return False, [f"A {wave}-survey is already recorded for this account."]
    except SQLAlchemyError:
        return False, ["Your answers could not be saved just now. Please try "
                       "again in a moment — they are still filled in."]
    return True, []


def enrolment_counts() -> dict[str, int]:
    """Aggregate numbers for the researcher panel. No identities."""
    engine = db.get_engine()
    counts = {"agreed": 0, "declined": 0, "withdrawn": 0, "pre": 0, "post": 0}
    if engine is None:
        return counts
    with engine.connect() as connection:
        for decision, withdrawn, n in connection.execute(
                select(db.consents.c.decision,
                       db.consents.c.withdrawn_at.is_not(None),
                       func.count())
                .group_by(db.consents.c.decision,
                          db.consents.c.withdrawn_at.is_not(None))):
            counts["withdrawn" if withdrawn else decision] += n
        for wave, n in connection.execute(
                select(db.survey_responses.c.wave, func.count())
                .group_by(db.survey_responses.c.wave)):
            counts[wave] = n
    return counts
