# app/study/flow.py
"""Where a person is in the study, and what that allows. No I/O.

    undecided ─┬─ agree ───▶ pre_due ─▶ waiting ─▶ post_due ─▶ complete
               └─ decline ─▶ declined ── agree later ──▶ pre_due

    any agreed stage ── withdraw ──▶ withdrawn   (final)

Two rules carry the design:

* **The baseline comes before any AI feedback.** A student teacher cannot score
  a plan until they have either declined, or agreed and answered the
  pre-survey. Otherwise the "pre" measure could be taken after exposure to the
  very thing it is a baseline for.
* **Declining costs nothing.** A decline unblocks the dashboard immediately and
  is never asked about again, so the only people the gate holds up are those
  who chose to take part — and they can withdraw at any time.

Exposure between the waves is counted in plans, not days: the post-survey opens
once ``POST_AFTER_SUBMISSIONS`` plans have been scored *since* the pre-survey.
Counted from the pre-survey rather than from zero so that someone who declined,
used the tool for weeks, then opted in still sees N plans' worth of feedback
before their follow-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

#: Accounts that are invited to take part. Tutors and researchers use the same
#: dashboard but are not the population under study.
PARTICIPANT_ROLES = frozenset({"student_teacher"})

POST_AFTER_SUBMISSIONS = int(os.getenv("COTEACH_POST_AFTER_SUBMISSIONS", 2))

NOT_PARTICIPANT = "not_participant"
UNAVAILABLE = "unavailable"
UNDECIDED = "undecided"
DECLINED = "declined"
WITHDRAWN = "withdrawn"
PRE_DUE = "pre_due"
WAITING = "waiting"
POST_DUE = "post_due"
COMPLETE = "complete"


@dataclass(frozen=True)
class ConsentRecord:
    decision: str                      # "agreed" | "declined"
    participant_code: Optional[str] = None
    withdrawn: bool = False

    @property
    def agreed(self) -> bool:
        return self.decision == "agreed" and not self.withdrawn


@dataclass(frozen=True)
class StudyState:
    stage: str
    participant_code: Optional[str] = None
    plans_since_pre: int = 0

    @property
    def upload_blocked(self) -> bool:
        return self.stage in (UNDECIDED, PRE_DUE)

    @property
    def due_wave(self) -> Optional[str]:
        return {PRE_DUE: "pre", POST_DUE: "post"}.get(self.stage)

    @property
    def plans_until_post(self) -> int:
        return max(0, POST_AFTER_SUBMISSIONS - self.plans_since_pre)

    @property
    def block_reason(self) -> str:
        if self.stage == UNDECIDED:
            return ("Before you upload a plan, please choose whether to take "
                    "part in the research study (see the panel above).")
        if self.stage == PRE_DUE:
            return ("You agreed to take part, so please answer the short "
                    "first survey in the **Trust survey** tab before your "
                    "first plan is scored. It has to come before you see any "
                    "AI feedback.")
        return ""


def resolve(role: str, storage_ok: bool, consent: Optional[ConsentRecord],
            completed: Mapping[str, int], submissions: int) -> StudyState:
    """The study state for one person.

    *completed* maps each answered wave to the number of plans that had been
    scored when it was answered; *submissions* is the current total.
    """
    if role not in PARTICIPANT_ROLES:
        return StudyState(NOT_PARTICIPANT)
    if not storage_ok:
        # Nothing can be recorded, so nothing is asked and nothing is blocked.
        return StudyState(UNAVAILABLE)
    if consent is None:
        return StudyState(UNDECIDED)
    if consent.withdrawn:
        return StudyState(WITHDRAWN, consent.participant_code)
    if consent.decision != "agreed":
        return StudyState(DECLINED)

    code = consent.participant_code
    if "pre" not in completed:
        return StudyState(PRE_DUE, code)

    since = max(0, submissions - completed["pre"])
    if "post" in completed:
        return StudyState(COMPLETE, code, since)
    if since >= POST_AFTER_SUBMISSIONS:
        return StudyState(POST_DUE, code, since)
    return StudyState(WAITING, code, since)
