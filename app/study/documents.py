# app/study/documents.py
"""The participant-facing documents, read from ``docs/irb/``.

The dashboard shows participants the same files that go to the ethics
committee. Keeping a second copy of the consent wording in Python would let
the approved text and the text people actually agreed to drift apart, and
nobody would notice until an audit.

Unfilled details are written ``{{LIKE_THIS}}``. ``unresolved_placeholders()``
finds them, and ``approval.approval_problems()`` refuses to call the materials
approved while any remain.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "irb"

INFORMATION_SHEET = DOCS_DIR / "participant_information_sheet.md"
CONSENT_FORM = DOCS_DIR / "consent_form.md"

#: The documents a participant sees before agreeing. Their text is part of the
#: materials fingerprint; the data-management plan and the README are not.
PARTICIPANT_DOCUMENTS = (INFORMATION_SHEET, CONSENT_FORM)

PLACEHOLDER = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")
_STATEMENT = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")


def read(path: Path) -> str:
    """File text with line endings normalised, or "" if it is missing.

    Normalised because a checkout with ``core.autocrlf`` would otherwise give
    the same document a different fingerprint on Windows.
    """
    try:
        return path.read_text(encoding="utf-8").replace("\r\n", "\n")
    except FileNotFoundError:
        return ""


def body(path: Path) -> str:
    """The document without its top-level title, for display under our own."""
    lines = read(path).split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def consent_statements() -> list[str]:
    """The numbered statements a participant confirms by agreeing."""
    return [match.group(1) for line in read(CONSENT_FORM).split("\n")
            if (match := _STATEMENT.match(line))]


def unresolved_placeholders() -> dict[str, list[str]]:
    """``{file name: [placeholder, ...]}`` for every participant document."""
    found: dict[str, list[str]] = {}
    for path in PARTICIPANT_DOCUMENTS:
        names = sorted(set(PLACEHOLDER.findall(read(path))))
        if names:
            found[path.name] = names
    return found


def missing_documents() -> list[str]:
    return [path.name for path in PARTICIPANT_DOCUMENTS if not path.exists()]
