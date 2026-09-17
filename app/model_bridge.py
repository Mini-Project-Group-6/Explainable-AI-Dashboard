# app/model_bridge.py
"""Locate and import S1/S2's model package. S3's side of the boundary, only.

This module deliberately declares **nothing** about the rubric. The criteria,
the feature order, the score scale, the band thresholds and the explanation
colours all live in ``model_contract`` / ``rubric_schema`` /
``explainability.*``, which state plainly that everything downstream imports
through them and never restates them. A second copy in the dashboard would be
a second source of truth, and the one it would drift from is the one the models
were actually trained against.

So all this does is:

  * put ``model/`` on ``sys.path`` (S1's package imports ``model_contract``,
    ``compat`` etc. as top-level modules, so the package root goes on the path,
    not the repo root);
  * import it once and hold the result;
  * report *why* it is unavailable when it is, so the dashboard can say so
    instead of rendering an empty page.

When the package cannot be imported the dashboard degrades to a status screen.
It does not substitute placeholder scores: a plausible-looking SHAP panel with
invented numbers is precisely the failure this project exists to measure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"


@dataclass(frozen=True)
class ModelLayer:
    """The imported model package, or the reason there isn't one."""

    contract: Optional[ModuleType] = None
    render: Optional[ModuleType] = None
    predict: Optional[ModuleType] = None
    error: str = ""
    hint: str = ""

    @property
    def available(self) -> bool:
        return self.contract is not None and self.render is not None


@lru_cache(maxsize=1)
def model_layer() -> ModelLayer:
    """Import S1/S2's package once per process.

    Cached rather than retried: the import pulls in torch and spaCy and takes
    seconds, and Streamlit re-runs the script on every interaction.
    """
    if not (MODEL_ROOT / "model_contract.py").is_file():
        return ModelLayer(
            error=f"No model package at {MODEL_ROOT}.",
            hint="S1/S2's code lives on feature/cactus-jack. Merge it into this "
                 "branch (git merge feature/cactus-jack) to enable scoring.")

    if str(MODEL_ROOT) not in sys.path:
        sys.path.insert(0, str(MODEL_ROOT))

    try:
        import model_contract
        from explainability import render
    except Exception as error:                        # noqa: BLE001
        # Covers ArtifactContractError (a trained bundle that no longer matches
        # the contract) as well as a missing dependency. Both are actionable by
        # a human and neither should take the dashboard down.
        return ModelLayer(
            error=f"{type(error).__name__}: {error}",
            hint="Install the model requirements (pip install -r "
                 "model/requirements.txt) and check the artifacts match the "
                 "current contract version.")

    predict: Optional[ModuleType] = None
    try:
        from scoring import predict as predict_module
        predict = predict_module
    except Exception as error:                        # noqa: BLE001
        # Explanations can still be rendered from a stored result even when the
        # scorer itself will not load, so this is not fatal.
        return ModelLayer(contract=model_contract, render=render,
                          error=f"Scorer unavailable — {type(error).__name__}: {error}",
                          hint="Uploads cannot be scored until this is resolved.")

    return ModelLayer(contract=model_contract, render=render, predict=predict)


def artifacts_status() -> dict[str, object]:
    """What S1 has left on disk, for the transparency panel.

    Paths come from the contract when it is importable, so this never hardcodes
    an artifacts location that S1 could change.
    """
    layer = model_layer()
    if layer.contract is not None:
        artifacts_dir = Path(layer.contract.ARTIFACTS_DIR)
        bundle = Path(layer.contract.RUBRIC_MODEL_PATH)
        text_dir = Path(layer.contract.TEXT_MODEL_DIR)
    else:
        artifacts_dir = MODEL_ROOT / "artifacts"
        bundle = artifacts_dir / "rubric_model.joblib"
        text_dir = artifacts_dir / "bert_rubric"
    return {
        "artifacts_dir": str(artifacts_dir),
        "rubric_model": bundle.is_file(),
        "text_model": text_dir.is_dir(),
    }
