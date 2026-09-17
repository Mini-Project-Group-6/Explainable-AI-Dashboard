# app/study/approval.py
"""Whether the study materials are approved for data collection.

Approval is pinned to content, not to a flag. ``is_approved()`` is true only
when a protocol number is recorded *and* ``APPROVED_FINGERPRINT`` matches the
current items plus the participant documents. Change one word of an approved
statement and the survey drops back to draft by itself — the same idea as
``model_contract`` refusing a model fitted to a different corpus.

Until then the survey still runs, so the team can pilot it, but every response
is stored with ``irb_protocol = NULL`` and the export leaves those rows out
unless asked to include them.

Recording approval, once the committee has signed off:

    python -m app.study status          # prints the current fingerprint

then set both constants below to the protocol number and that fingerprint.
Do not put the protocol number inside the participant documents: that would
change the fingerprint the committee approved. The dashboard shows it from
``IRB_PROTOCOL`` instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from app.study import documents, flow, instruments

#: The ethics committee's reference for this study, once approved.
IRB_PROTOCOL: Optional[str] = None

#: ``fingerprint()`` of the exact materials the committee approved.
APPROVED_FINGERPRINT: Optional[str] = None


def fingerprint() -> str:
    """A short hash of the items, the participant documents and the procedure.

    The procedure is in because participants are told it: the information
    sheet says the follow-up opens after two scored plans, so changing that
    number changes what was approved just as much as rewording an item.
    """
    payload = {
        "instruments": instruments.canonical(),
        "documents": {path.name: documents.read(path)
                      for path in documents.PARTICIPANT_DOCUMENTS},
        "procedure": {
            "participant_roles": sorted(flow.PARTICIPANT_ROLES),
            "post_after_submissions": flow.POST_AFTER_SUBMISSIONS,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def approval_problems() -> list[str]:
    """Everything standing between the current materials and approved status."""
    problems: list[str] = []

    missing = documents.missing_documents()
    if missing:
        problems.append(f"Participant documents missing: {', '.join(missing)}.")

    for name, placeholders in documents.unresolved_placeholders().items():
        problems.append(f"{name} still has placeholders to fill in: "
                        f"{', '.join(placeholders)}.")

    unverified = instruments.unverified_sources()
    if unverified:
        problems.append("Item wording not yet checked against the published "
                        f"source: {', '.join(unverified)}.")

    current = fingerprint()
    if IRB_PROTOCOL is None:
        problems.append("No ethics approval is recorded.")
    elif APPROVED_FINGERPRINT != current:
        problems.append(
            f"The materials have changed since approval "
            f"(approved {APPROVED_FINGERPRINT}, now {current}). Changed "
            f"wording needs the committee's approval before it is used.")
    return problems


def is_approved() -> bool:
    return not approval_problems()


def protocol_for_record() -> Optional[str]:
    """What to store on a response: the protocol if approved, else ``None``."""
    return IRB_PROTOCOL if is_approved() else None
