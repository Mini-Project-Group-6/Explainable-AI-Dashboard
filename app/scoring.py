# app/scoring.py
"""Upload -> S1's scorer -> S2's view model. S3 owns the plumbing, nothing else.

No score, SHAP value or explanation sentence originates in this module. Its
whole job is lifecycle and I/O around ``scoring.predict.RubricScorer``:

  * load the scorer once per server process (it reads S1's artifacts and a
    DistilBERT checkpoint — far too expensive for Streamlit's per-interaction
    re-run);
  * spool an uploaded file to disk, because ``score_file`` takes a path;
  * cache the result per submission, so switching tabs does not re-score;
  * hand S2's renderer the objects it expects.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from app.model_bridge import model_layer

#: File types S1's ingestion layer can read. ``.txt`` goes through the raw-text
#: path rather than the document extractor.
ACCEPTED_TYPES = ["pdf", "docx", "txt"]


@st.cache_resource(show_spinner=False)
def get_scorer():
    """S1's ``RubricScorer``, loaded once. ``None`` if the model layer is absent.

    ``with_text_attributions`` stays off. S1 measured span attribution at ~93s
    against a <=15s budget, documented that the cost is irreducible, and left
    the instruction that S3 precompute and cache it per submission rather than
    let it block an upload. The tabular channel explains all ten criteria on its
    own, so withholding it costs coverage nothing.
    """
    layer = model_layer()
    if layer.predict is None:
        return None
    return layer.predict.RubricScorer(
        with_shap=True, with_text=True, with_text_attributions=False)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """S1's validated ``Artifacts`` handle, for the transparency panel.

    Deliberately not ``get_scorer().artifacts``: the panel only needs to report
    which artifacts are on disk, while building a ``RubricScorer`` also loads
    DistilBERT and ten SHAP explainers. Tabs render on every page load, so
    reading it from the scorer would put a multi-second model load in front of
    every visitor, including those who never upload anything.
    """
    layer = model_layer()
    if layer.contract is None:
        return None
    try:
        return layer.contract.load_artifacts()
    except Exception:                                 # noqa: BLE001
        return None


def submission_id(uploaded_file) -> str:
    """Content hash, so re-uploading the same plan reuses the cached result."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]


def _score_bytes(key: str, name: str, payload: bytes) -> dict[str, Any]:
    """Score raw upload bytes.

    Not wrapped in ``st.cache_data``: the result carries S2's ``PlanExplanation``
    and ``Suggestion`` dataclasses, and that cache round-trips its values
    through pickle. Caching by content hash in session state (see
    ``score_upload``) gets the same dedupe without serialising S2's objects.
    """
    scorer = get_scorer()
    if scorer is None:
        raise RuntimeError("The model layer is not available.")
    layer = model_layer()

    # mkstemp rather than a predictable "coteach_<hash>" in the shared temp
    # directory: on a multi-user deployment that path is guessable, and it
    # holds a student teacher's lesson plan. mkstemp creates it 0600 with a
    # random name and no chance of another process winning the race.
    handle, raw_path = tempfile.mkstemp(suffix=Path(name).suffix or ".pdf",
                                        prefix="coteach_")
    os.close(handle)
    path = Path(raw_path)
    path.write_bytes(payload)
    try:
        if path.suffix.lower() == ".txt":
            result = scorer.score_text(path.read_text(encoding="utf-8"), name)
        else:
            result = scorer.score_file(path)
    finally:
        path.unlink(missing_ok=True)

    view = layer.render.plan_view(result.explanation, result.suggestions)
    view["source"] = name
    return {
        "view": view,
        "explanation": result.explanation,
        "suggestions": result.suggestions,
        "features": result.features,
        "rubric_scores": result.rubric_scores,
        "overall_score_0_100": result.overall_score_0_100,
        "missing_sections": result.missing_sections,
        "latency_seconds": result.latency_seconds,
        "used_text_channel": result.used_text_channel,
        "used_text_attributions": result.used_text_attributions,
    }


def score_upload(uploaded_file) -> dict[str, Any]:
    """Score an uploaded lesson plan, reusing the result for an identical file.

    Streamlit re-runs the script on every interaction and hands back the same
    uploaded file each time, so without this a click on any tab would re-score
    the plan. Keyed on content, not filename: a renamed but identical plan is
    the same submission, and an edited plan under the same name is not.
    """
    key = submission_id(uploaded_file)
    cache = st.session_state.setdefault("scored_submissions", {})
    if key not in cache:
        cache[key] = _score_bytes(key, uploaded_file.name,
                                  uploaded_file.getvalue())
    return cache[key]


def shap_explanations(features: dict[str, float]) -> dict[str, Any]:
    """Per-criterion SHAP ``Explanation`` objects for one plan's features.

    S2's ``st_waterfall`` and ``st_force_plot`` take a raw shap Explanation, but
    ``ScoringResult`` flattens those into plain lists, so there is no route from
    a scored submission to the plots. Re-deriving them from S1's own explainer
    keeps SHAP out of S3 entirely — the object is built by S2's
    ``RubricExplainer`` and handed straight to S2's renderer.

    Cheap enough to redo per interaction: TreeExplainer over 22 features is
    milliseconds, against the ~1.2s the text channel already costs.
    """
    scorer = get_scorer()
    if scorer is None or getattr(scorer, "explainer", None) is None:
        return {}
    return scorer.explainer.explain_one(features)


def scorer_error() -> Optional[str]:
    """Why scoring is unavailable, or ``None`` when it is ready."""
    layer = model_layer()
    if layer.predict is not None:
        return None
    return layer.error or "The model layer is not available."
