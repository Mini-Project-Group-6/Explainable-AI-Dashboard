"""``python -m app.study`` — the study's command line. See ``app/study/__init__.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.study import approval, documents, export, instruments, store

INSTRUMENTS_DOC = documents.DOCS_DIR / "survey_instruments.md"


def _status() -> int:
    problems = approval.approval_problems()
    print(f"Instrument version     {instruments.INSTRUMENT_VERSION}")
    print(f"Materials fingerprint  {approval.fingerprint()}")
    print(f"Recorded approval      {approval.IRB_PROTOCOL or 'none'}"
          + (f" (fingerprint {approval.APPROVED_FINGERPRINT})"
             if approval.IRB_PROTOCOL else ""))
    print()
    if problems:
        print("NOT APPROVED — responses are stored as pilot data:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("Approved. Responses are stored as study data.")
    print()
    counts = store.enrolment_counts()
    print("Enrolment  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def _instruments(write: bool) -> int:
    text = instruments.adaptation_markdown()
    if not write:
        print(text)
        return 0
    INSTRUMENTS_DOC.parent.mkdir(parents=True, exist_ok=True)
    INSTRUMENTS_DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {INSTRUMENTS_DOC}")
    return 0


def _export(out: Path, include_pilot: bool) -> int:
    tables = export.build(include_pilot=include_pilot)
    if not tables["participants"][1]:
        print("Nothing to export: no participants"
              + ("" if include_pilot else
                 " consented under approved materials. Add --include-pilot "
                 "to export pilot data."))
        return 1
    for path in export.write(out, tables=tables):
        print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # The Windows console defaults to cp1252, which cannot print the dashes
    # and arrows in the generated appendix.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m app.study")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="approval state, fingerprint and enrolment")
    shown = sub.add_parser("instruments", help="the instrument appendix")
    shown.add_argument("--write", action="store_true",
                       help=f"write it to {INSTRUMENTS_DOC.relative_to(documents.REPO_ROOT)}")
    out = sub.add_parser("export", help="anonymised CSVs for analysis")
    out.add_argument("--out", type=Path, required=True)
    out.add_argument("--include-pilot", action="store_true",
                     help="include data collected before approval")
    args = parser.parse_args(argv)

    if args.command == "status":
        return _status()
    if args.command == "instruments":
        return _instruments(args.write)
    return _export(args.out, args.include_pilot)


if __name__ == "__main__":
    raise SystemExit(main())
