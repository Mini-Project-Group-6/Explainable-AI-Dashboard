"""``python -m analysis`` — run the S5 analysis. See ``analysis/__init__.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis import load as loader
from analysis import report, stats
from analysis.synthetic import write_synthetic_export
from app.database.db import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "analysis"


def _run(export_dir: Path, out_dir: Path, include_pilot: bool,
         include_late_baseline: bool) -> int:
    data = loader.load(export_dir, include_pilot=include_pilot,
                       include_late_baseline=include_late_baseline)
    results = report.analyse(data)
    written = report.write(data, results, out_dir)

    if data.synthetic:
        print("SYNTHETIC DATA — these results describe no real participants.")
    print(f"Pre/post pairs analysed: {data.flow['paired_analysed']} "
          f"(of {data.flow['participants_analysed']} participants)")
    primary = results.prepost.set_index("scale").loc[report.PRIMARY_SCALE]
    print(f"Trust {stats.fmt(primary['mean_pre'])} -> "
          f"{stats.fmt(primary['mean_post'])}, "
          f"t({primary['df']}) = {stats.fmt(primary['t'])}, "
          f"{stats.p_clause(primary['p'])}, d_z = {stats.fmt(primary['d_z'])}")
    for note in data.notes:
        print(f"Note: {note}")
    print(f"Wrote {len(written)} files; start with {out_dir / 'summary.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse an export")
    run.add_argument("--export", type=Path, required=True,
                     help="folder written by python -m app.study export")
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run.add_argument("--include-pilot", action="store_true",
                     help="include data collected before ethics approval")
    run.add_argument("--include-late-baseline", action="store_true",
                     help="keep first surveys answered after AI feedback in "
                          "the pre/post comparisons")

    fake = sub.add_parser("synthetic",
                          help="generate a synthetic export and analyse it")
    fake.add_argument("--out", type=Path,
                      default=REPO_ROOT / "data" / "analysis-synthetic")
    fake.add_argument("--n", type=int, default=60)
    fake.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args.export, args.out, args.include_pilot,
                        args.include_late_baseline)
        export_dir = write_synthetic_export(args.out / "export", n=args.n,
                                            seed=args.seed)
        return _run(export_dir, args.out / "report", False, False)
    except loader.AnalysisError as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
