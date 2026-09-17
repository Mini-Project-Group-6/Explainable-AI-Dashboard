# analysis/load.py
"""Read and screen the export written by ``python -m app.study export``.

Scale membership and reverse keying come from ``app.study.instruments`` — the
same module that asked the questions — rather than a second copy here. Scale
means are recomputed from the raw items and compared with the export's own
``*_mean`` columns, so a mismatch between collection and analysis shows up in
the report instead of in the results.

Screening, applied in this order and counted in the sample flow:

1. **Pilot data** — responses without a protocol number, and participants who
   consented to unapproved materials — is dropped unless ``include_pilot``.
2. **Mixed instrument versions** are refused: items may differ between them.
3. **Late baselines** — a pre-survey answered after the participant had
   already seen AI feedback — are kept for descriptives but left out of the
   pre/post comparisons unless ``include_late_baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.study import export, instruments

SYNTHETIC_MARKER = "SYNTHETIC"


class AnalysisError(Exception):
    """The export cannot be analysed as it stands. Safe to print."""


@dataclass
class StudyData:
    participants: pd.DataFrame
    #: One row per participant × wave: raw item columns, then ``<scale>`` means
    #: computed here from reverse-keyed items.
    responses: pd.DataFrame
    submissions: pd.DataFrame
    #: Participants whose pre-survey is excluded from pre/post comparisons.
    late_baseline: set[str]
    flow: dict[str, int]
    source: Path
    synthetic: bool
    include_pilot: bool
    include_late_baseline: bool
    instrument_version: str
    notes: list[str] = field(default_factory=list)


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise AnalysisError(f"Missing {path.name} in {path.parent}. Export first: "
                            f"python -m app.study export --out {path.parent}")
    frame = pd.read_csv(path, encoding="utf-8-sig",
                        dtype={"participant_code": str}, keep_default_na=True)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise AnalysisError(f"{path.name} lacks column(s) {', '.join(missing)}; "
                            f"it may come from a different export version.")
    return frame


def keyed_items(responses: pd.DataFrame, scale_key: str) -> pd.DataFrame:
    """The items of one scale with reverse-worded ones flipped."""
    members = [item for item in instruments.ITEMS if item.scale == scale_key]
    return pd.DataFrame({
        item.id: responses[item.id].map(
            lambda v, item=item: np.nan if pd.isna(v)
            else instruments.keyed(item, int(v)))
        for item in members
    }, index=responses.index)


def load(export_dir: Path, include_pilot: bool = False,
         include_late_baseline: bool = False) -> StudyData:
    export_dir = Path(export_dir)
    participants = _read(export_dir / "participants.csv", export.PARTICIPANT_COLUMNS)
    responses = _read(export_dir / "responses.csv", export.RESPONSE_COLUMNS)
    submissions = _read(export_dir / "submissions.csv", export.SUBMISSION_BASE_COLUMNS)
    notes: list[str] = []
    flow = {"participants_in_export": len(participants)}

    if not include_pilot:
        pilot_people = set(participants.loc[participants["pilot_consent"] == 1,
                                            "participant_code"])
        flow["excluded_pilot_consent"] = len(pilot_people)
        participants = participants[~participants["participant_code"].isin(pilot_people)]
        before = len(responses)
        responses = responses[(responses["pilot"] != 1)
                              & ~responses["participant_code"].isin(pilot_people)]
        flow["excluded_pilot_responses"] = before - len(responses)
        submissions = submissions[~submissions["participant_code"].isin(pilot_people)]

    versions = sorted(responses["instrument_version"].dropna().astype(str).unique())
    if len(versions) > 1:
        raise AnalysisError(f"Responses come from instrument versions "
                            f"{', '.join(versions)}. Analyse each version "
                            f"separately; their items may differ.")
    version = versions[0] if versions else instruments.INSTRUMENT_VERSION
    if version != instruments.INSTRUMENT_VERSION:
        raise AnalysisError(
            f"The export was collected with instrument {version}, but this code "
            f"defines {instruments.INSTRUMENT_VERSION}. Check out the version of "
            f"app/study/instruments.py that matches the data.")

    responses = responses.copy()
    mismatched = 0
    for scale in instruments.SCALES:
        means = keyed_items(responses, scale.key).mean(axis=1, skipna=False)
        exported = pd.to_numeric(responses[f"{scale.key}_mean"], errors="coerce")
        both = means.notna() & exported.notna()
        # The export rounds means to 4 decimal places.
        mismatched += int((np.abs(means[both] - exported[both]) > 1e-3).sum())
        responses[scale.key] = means
    if mismatched:
        notes.append(f"{mismatched} exported scale mean(s) differ from the means "
                     f"recomputed from items; the recomputed values are used.")

    pre = responses[responses["wave"] == "pre"]
    late = set(pre.loc[pd.to_numeric(pre["pre_after_feedback"], errors="coerce") == 1,
                       "participant_code"])
    flow["pre_completed"] = int(pre["participant_code"].nunique())
    flow["post_completed"] = int(
        responses.loc[responses["wave"] == "post", "participant_code"].nunique())
    both = (set(pre["participant_code"])
            & set(responses.loc[responses["wave"] == "post", "participant_code"]))
    flow["both_waves"] = len(both)
    flow["late_baseline"] = len(late & both)
    flow["paired_analysed"] = len(both if include_late_baseline else both - late)
    flow["participants_analysed"] = len(participants)

    return StudyData(
        participants=participants.reset_index(drop=True),
        responses=responses.reset_index(drop=True),
        submissions=submissions.reset_index(drop=True),
        late_baseline=set() if include_late_baseline else late,
        flow=flow,
        source=export_dir,
        synthetic=(export_dir / SYNTHETIC_MARKER).exists(),
        include_pilot=include_pilot,
        include_late_baseline=include_late_baseline,
        instrument_version=version,
        notes=notes,
    )
