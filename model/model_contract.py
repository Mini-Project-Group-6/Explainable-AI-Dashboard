"""The single source of truth binding S1's trained artifacts to S2's explanations.

Everything downstream — SHAP plots, the reconciled per-criterion explanation,
the revision suggestions, S3's dashboard — imports through here and never
reaches for a model file directly.

What this module declares:

1. Artifact paths and the contract version they must satisfy.
2. ``FEATURE_ORDER`` — the frozen feature names in *training* order (22 as of
   contract v2.0.0; read ``N_FEATURES``, never a literal). This is an
   independent copy, not a re-export: it is compared against the live extractor
   at import time so a silent reorder/rename in
   ``ingestion/feature_engineer.py`` fails fast instead of quietly producing
   wrong SHAP attributions. (Feature order has broken before — see DEV.md.)
3. ``CRITERIA`` — the ten plan-assessable rubric criteria, each with a stable
   ID, the schema key used by the models, a tutor-readable label, and its NTS
   indicator.
4. ``OUTPUT_SHAPES`` — what XGBoost, SHAP and DistilBERT are each expected to
   return, so a shape regression is caught at the boundary.
5. ``load_artifacts()`` — the one loader, validating all of the above.

Scope note: the rubric is ten *plan-assessable* criteria. The STS School
Placement Handbook checklist has 25 items; the other 15 assess live teaching
delivery and cannot be judged from a written plan. That exclusion is
deliberate and documented — do not add criteria here that require observing a
lesson being taught.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional

from compat import preload_torch
from rubric_schema import RUBRIC_DIMENSIONS, RUBRIC_WEIGHTS

# Must run before joblib unpickles an XGBoost model, or torch can no longer
# initialise in this process. See compat.py — this is not optional on Windows.
preload_torch()

# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------

#: Bumped whenever anything below changes in a way that invalidates a trained
#: bundle: feature order/names, criterion keys, or the score scale. A bundle
#: trained under a different major version must be retrained, not coerced.
#: 2.0.0 — added F19-F22 so every rubric criterion has at least one feature that
#: measures it. This changes the frozen feature order, so every bundle trained
#: under 1.x is invalid and must be retrained; ``validate_bundle`` will reject
#: them rather than let SHAP index into the wrong columns.
CONTRACT_VERSION: Final[str] = "2.0.0"

#: Rubric criterion scores are ordinal 1-4 (see rubric_schema.overall_score,
#: which maps the weighted mean onto 0-100). Predictions are clipped to this.
SCORE_MIN: Final[int] = 1
SCORE_MAX: Final[int] = 4


# --------------------------------------------------------------------------
# Artifact paths
# --------------------------------------------------------------------------

MODEL_ROOT: Final[Path] = Path(__file__).resolve().parent

#: Overridable so S3's deployment can point at a mounted volume without
#: editing code. Defaults to ``model/artifacts/`` (gitignored — regenerable).
ARTIFACTS_DIR: Final[Path] = Path(
    os.environ.get("COTEACH_ARTIFACTS_DIR", MODEL_ROOT / "artifacts")
).resolve()

#: joblib bundle written by ``scoring/train_xgboost.py``.
RUBRIC_MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "rubric_model.joblib"

#: HuggingFace-format directory for the LoRA-adapted DistilBERT text scorer.
#: Optional: absent until S1 produces it, and every consumer here degrades to
#: tabular-only rather than failing.
TEXT_MODEL_DIR: Final[Path] = ARTIFACTS_DIR / "bert_rubric"

#: Corpus-level SHAP plots (beeswarm/waterfall PNGs) from explainability/shap_tree.py.
SHAP_PLOT_DIR: Final[Path] = ARTIFACTS_DIR / "shap"


# --------------------------------------------------------------------------
# Frozen feature order
# --------------------------------------------------------------------------

#: The structural features in the exact order the XGBoost models were trained
#: on, which is also the column order SHAP indexes into. Declared independently
#: of the extractor on purpose — see module docstring. F19-F22 were added in
#: contract v2.0.0; use ``N_FEATURES`` wherever the count is needed.
FEATURE_ORDER: Final[tuple[str, ...]] = (
    "objective_count",                 # F1
    "smart_objective_count",           # F2
    "objective_measurability_ratio",   # F3
    "bloom_remember_prop",             # F4
    "bloom_understand_prop",           # F5
    "bloom_apply_prop",                # F6
    "bloom_analyze_prop",              # F7
    "bloom_evaluate_prop",             # F8
    "bloom_create_prop",               # F9
    "content_activity_ratio",          # F10
    "learner_activity_verb_density",   # F11
    "activity_variety_count",          # F12
    "assessment_alignment_score",      # F13
    "assessment_item_count",           # F14
    "time_allocation_coverage",        # F15
    "time_total_consistency",          # F16
    "sentence_complexity_index",       # F17 (z-scored at train time)
    "readability_flesch",              # F18
    "resource_specificity_count",      # F19
    "rpk_link_score",                  # F20
    "differentiation_strategy_count",  # F21
    "closure_quality_score",           # F22
)

N_FEATURES: Final[int] = len(FEATURE_ORDER)

#: F17 is stored raw by the extractor and z-scored using statistics frozen into
#: the bundle. Inference must apply the identical transform or SHAP values are
#: computed off-manifold.
Z_SCORED_FEATURES: Final[tuple[str, ...]] = ("sentence_complexity_index",)


# --------------------------------------------------------------------------
# Rubric criteria
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterion:
    """One plan-assessable rubric criterion."""

    id: str                 # stable display/report ID, e.g. "C01"
    key: str                # schema key used by the models and labels.csv
    label: str              # tutor-readable name
    nts_indicator: str      # NTS code, e.g. "3e"
    nts_descriptor: str     # the NTS wording the code refers to
    weight: float           # rubric weight, from rubric_schema.RUBRIC_WEIGHTS
    plan_evidence: str      # what a tutor looks for in the written plan

    @property
    def nts_verified(self) -> bool:
        """True once the code has been checked against the handbook."""
        return self.nts_indicator in _VERIFIED_NTS_CODES


#: NTS indicator codes below are matched by *descriptor wording* against the
#: National Teachers' Standards and still need one pass against the printed
#: STS School Placement Handbook to confirm the letter suffixes. Add a code to
#: this set once a human has confirmed it; ``unverified_criteria()`` reports
#: the rest. The descriptors are the reliable anchor — match on those.
_VERIFIED_NTS_CODES: Final[frozenset[str]] = frozenset()

NTS_SOURCE: Final[str] = (
    "National Teachers' Standards for Ghana (2017), as referenced by the STS "
    "School Placement Handbook lesson observation checklist"
)

CRITERIA: Final[tuple[Criterion, ...]] = (
    Criterion(
        id="C01",
        key="learning_outcomes",
        label="Learning outcomes",
        nts_indicator="3a",
        nts_descriptor=("Plans and delivers varied and challenging lessons, showing "
                        "clear grasp of the intended outcomes of their teaching"),
        weight=RUBRIC_WEIGHTS["learning_outcomes"],
        plan_evidence="Objectives are specific, measurable and stated as learner outcomes.",
    ),
    Criterion(
        id="C02",
        key="pedagogical_content_knowledge",
        label="Pedagogical content knowledge",
        nts_indicator="2c",
        nts_descriptor=("Has secure content knowledge, pedagogical knowledge and "
                        "pedagogical content knowledge for the school and grade taught"),
        weight=RUBRIC_WEIGHTS["pedagogical_content_knowledge"],
        plan_evidence="Core points are accurate, pitched to the grade, and anticipate misconceptions.",
    ),
    Criterion(
        id="C03",
        key="teaching_learning_strategies",
        label="Teaching and learning strategies",
        nts_indicator="3e",
        nts_descriptor=("Employs a variety of instructional strategies that encourage "
                        "learner participation and critical thinking"),
        weight=RUBRIC_WEIGHTS["teaching_learning_strategies"],
        plan_evidence="Varied, learner-centred activities rather than continuous exposition.",
    ),
    Criterion(
        id="C04",
        key="resources_including_ict",
        label="Resources including ICT",
        nts_indicator="3j",
        nts_descriptor=("Produces and uses a variety of teaching and learning resources, "
                        "including ICT, to enhance learning"),
        weight=RUBRIC_WEIGHTS["resources_including_ict"],
        plan_evidence="Named, lesson-specific TLMs and any ICT use.",
    ),
    Criterion(
        id="C05",
        key="assessment_strategies_in_plan",
        label="Assessment strategies in the plan",
        nts_indicator="3k",
        nts_descriptor=("Integrates a variety of assessment modes into teaching to "
                        "support learning"),
        weight=RUBRIC_WEIGHTS["assessment_strategies_in_plan"],
        plan_evidence="Assessment items are present and test what the objectives promised.",
    ),
    Criterion(
        id="C06",
        key="lesson_introduction_rpk",
        label="Lesson introduction and RPK",
        nts_indicator="3a",
        nts_descriptor=("Plans and delivers varied and challenging lessons, showing "
                        "clear grasp of the intended outcomes of their teaching"),
        weight=RUBRIC_WEIGHTS["lesson_introduction_rpk"],
        plan_evidence="Starter connects relevant previous knowledge to the new topic.",
    ),
    Criterion(
        id="C07",
        key="lesson_sequencing",
        label="Lesson sequencing and timing",
        nts_indicator="3a",
        nts_descriptor=("Plans and delivers varied and challenging lessons, showing "
                        "clear grasp of the intended outcomes of their teaching"),
        weight=RUBRIC_WEIGHTS["lesson_sequencing"],
        plan_evidence="Starter/main/plenary all present, ordered, and realistically timed.",
    ),
    Criterion(
        id="C08",
        key="attention_to_all_learners",
        label="Attention to all learners",
        nts_indicator="3f",
        nts_descriptor=("Pays attention to all learners, especially girls and learners "
                        "with Special Educational Needs, ensuring their progress"),
        weight=RUBRIC_WEIGHTS["attention_to_all_learners"],
        plan_evidence="Differentiated tasks, SEN support, and equitable participation.",
    ),
    Criterion(
        id="C09",
        key="concept_explanation_examples",
        label="Concept explanation and examples",
        nts_indicator="3i",
        nts_descriptor=("Explains concepts clearly using examples familiar to learners"),
        weight=RUBRIC_WEIGHTS["concept_explanation_examples"],
        plan_evidence="Explanation strategy uses analogies/examples from the learners' context.",
    ),
    Criterion(
        id="C10",
        key="lesson_closure",
        label="Lesson closure",
        nts_indicator="3a",
        nts_descriptor=("Plans and delivers varied and challenging lessons, showing "
                        "clear grasp of the intended outcomes of their teaching"),
        weight=RUBRIC_WEIGHTS["lesson_closure"],
        plan_evidence="Plenary consolidates the indicator and checks attainment.",
    ),
)

CRITERION_KEYS: Final[tuple[str, ...]] = tuple(c.key for c in CRITERIA)
N_CRITERIA: Final[int] = len(CRITERIA)

_BY_KEY: Final[dict[str, Criterion]] = {c.key: c for c in CRITERIA}
_BY_ID: Final[dict[str, Criterion]] = {c.id: c for c in CRITERIA}


def criterion(key_or_id: str) -> Criterion:
    """Look up a criterion by schema key ("lesson_closure") or ID ("C10")."""
    found = _BY_KEY.get(key_or_id) or _BY_ID.get(key_or_id)
    if found is None:
        raise KeyError(
            f"unknown rubric criterion {key_or_id!r}; "
            f"expected one of {CRITERION_KEYS} or {tuple(_BY_ID)}")
    return found


def unverified_criteria() -> tuple[Criterion, ...]:
    """Criteria whose NTS code still needs checking against the handbook."""
    return tuple(c for c in CRITERIA if not c.nts_verified)


# --------------------------------------------------------------------------
# Expected output shapes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputSpec:
    """Expected shape/range of one model's output, for boundary checks."""

    name: str
    shape: tuple[str, ...]   # symbolic dims, e.g. ("n_plans", "n_criteria")
    dtype: str
    value_range: Optional[tuple[float, float]]
    notes: str


#: DistilBERT truncation length. Token attributions are only defined over the
#: tokens the model actually saw; anything past this is unattributed and the
#: reconciler must say so rather than imply the tail was considered.
TEXT_MAX_TOKENS: Final[int] = 512

OUTPUT_SHAPES: Final[dict[str, OutputSpec]] = {
    "xgboost_scores": OutputSpec(
        name="xgboost_scores",
        shape=("n_plans", "n_criteria"),
        dtype="float64",
        value_range=(float(SCORE_MIN), float(SCORE_MAX)),
        notes=("One regressor per criterion, columns in CRITERION_KEYS order. "
               "Rounded then clipped to [1, 4] for display; the unrounded value "
               "is what SHAP decomposes."),
    ),
    "shap_values": OutputSpec(
        name="shap_values",
        shape=("n_plans", "n_features"),
        dtype="float64",
        value_range=None,
        notes=("Per criterion. shap.Explanation with .values "
               "(n_plans, n_features), .base_values (n_plans,) and .data "
               "(n_plans, n_features). Columns follow FEATURE_ORDER; take the "
               "width from N_FEATURES, never a literal. Additive: "
               "base_value + values.sum() == the unrounded regressor output."),
    ),
    "shap_base_values": OutputSpec(
        name="shap_base_values",
        shape=("n_plans",),
        dtype="float64",
        value_range=None,
        notes="Expected value of the criterion's regressor over the training corpus.",
    ),
    "bert_scores": OutputSpec(
        name="bert_scores",
        shape=("n_plans", "n_criteria"),
        dtype="float32",
        value_range=(float(SCORE_MIN), float(SCORE_MAX)),
        notes=("DistilBERT+LoRA regression head, one output per criterion, same "
               "column order as xgboost_scores. NOT YET PRODUCED — "
               "scoring/train_bert_lora.py is a stub. Declared here so the text "
               "layer has a target to hit."),
    ),
    "bert_span_attributions": OutputSpec(
        name="bert_span_attributions",
        shape=("n_spans", "n_criteria"),
        dtype="float32",
        value_range=None,
        notes=("shap.Explainer (Partition/Owen values) over a Text masker, per "
               "plan. Spans are sentences or lines, not wordpieces, because that "
               "is the unit a tutor can be shown. Signed: positive raises the "
               "criterion score. Surfaced as reconcile.SpanAttribution with char "
               "offsets. NOT on the same scale as the tabular SHAP values — see "
               "explainability/reconcile.py for how the channels are compared."),
    ),
}

#: Text past this is truncated by the encoder and therefore unattributed; the
#: reconciler raises a caveat rather than letting the tail look considered.
TEXT_TRUNCATION_IS_CAVEATED: Final[bool] = True


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class ArtifactContractError(ValueError):
    """A trained artifact does not satisfy this contract. Never recoverable."""


#: Keys ``scoring/train_xgboost.train()`` must put in the bundle.
_REQUIRED_BUNDLE_KEYS: Final[tuple[str, ...]] = (
    "models", "feature_names", "rubric_dimensions", "f17_mean", "f17_std",
)


@dataclass(frozen=True)
class Artifacts:
    """A validated set of trained artifacts. The only handle downstream holds."""

    bundle: dict[str, Any]
    model_path: Path
    text_model_dir: Optional[Path]
    feature_names: tuple[str, ...] = FEATURE_ORDER
    criteria: tuple[Criterion, ...] = CRITERIA
    contract_version: str = CONTRACT_VERSION

    @property
    def has_text_model(self) -> bool:
        """False until S1 ships DistilBERT; consumers fall back to tabular-only."""
        return self.text_model_dir is not None

    @property
    def corpus_fingerprints(self) -> tuple[Optional[str], Optional[str]]:
        """(structural, text) training-corpus digests; None where unrecorded."""
        import json

        structural = self.bundle.get("corpus_fingerprint")
        text = None
        if self.text_model_dir is not None:
            manifest = self.text_model_dir / "text_bundle.json"
            if manifest.is_file():
                text = json.loads(
                    manifest.read_text(encoding="utf-8")).get("corpus_fingerprint")
        return structural, text

    @property
    def channels_consistent(self) -> bool:
        """True unless the two channels were demonstrably trained on different data.

        Unknown counts as consistent: artifacts predating the fingerprint have
        no digest to compare, and refusing to run on them would be worse than
        the risk. A *recorded mismatch* is the signal.
        """
        structural, text = self.corpus_fingerprints
        if structural is None or text is None:
            return True
        return structural == text

    @property
    def provenance_unverified(self) -> bool:
        """True when one channel records its corpus and the other does not.

        Not a mismatch — but not nothing either. A checkpoint carrying no
        fingerprint beside one that does is, by construction, older than the
        fingerprint itself, so it cannot be confirmed to share the corpus. That
        is worth saying out loud rather than passing silently as "consistent".
        """
        if not self.has_text_model:
            return False
        structural, text = self.corpus_fingerprints
        return (structural is None) != (text is None)

    def feature_index(self, name: str) -> int:
        """Column index of a feature — the index SHAP values are aligned to."""
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(f"{name!r} is not a contract feature") from None


def _mismatch_report(expected: tuple[str, ...], actual: tuple[str, ...]) -> str:
    """Human-readable diff, because 'lists differ' is useless at 3am."""
    lines = []
    missing = [n for n in expected if n not in actual]
    extra = [n for n in actual if n not in expected]
    if missing:
        lines.append(f"  missing from artifact: {missing}")
    if extra:
        lines.append(f"  unexpected in artifact: {extra}")
    if not missing and not extra:
        reordered = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        lines.append(f"  same names, REORDERED at positions: "
                     f"{[(i, f'expected {e}', f'got {a}') for i, e, a in reordered]}")
    return "\n".join(lines)


def validate_bundle(bundle: dict[str, Any], source: str = "<bundle>") -> None:
    """Raise ArtifactContractError unless *bundle* matches this contract.

    Checked separately from loading so tests and CI can validate an
    already-in-memory bundle.
    """
    if not isinstance(bundle, dict):
        raise ArtifactContractError(
            f"{source}: expected a dict bundle, got {type(bundle).__name__}")

    missing_keys = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing_keys:
        raise ArtifactContractError(
            f"{source}: bundle is missing required keys {missing_keys}. "
            f"Was it written by a different version of scoring/train_xgboost.py?")

    actual_features = tuple(bundle["feature_names"])
    if actual_features != FEATURE_ORDER:
        raise ArtifactContractError(
            f"{source}: feature names do not match contract "
            f"{CONTRACT_VERSION}.\n{_mismatch_report(FEATURE_ORDER, actual_features)}\n"
            "SHAP values are indexed positionally, so this would silently "
            "attribute one feature's contribution to another. Retrain against "
            "the current feature order — do not edit the bundle.")

    actual_criteria = tuple(bundle["rubric_dimensions"])
    if actual_criteria != CRITERION_KEYS:
        raise ArtifactContractError(
            f"{source}: rubric criteria do not match contract "
            f"{CONTRACT_VERSION}.\n{_mismatch_report(CRITERION_KEYS, actual_criteria)}")

    models = bundle["models"]
    absent = [k for k in CRITERION_KEYS if k not in models]
    if absent:
        raise ArtifactContractError(
            f"{source}: no trained model for criteria {absent}")

    for stat in ("f17_mean", "f17_std"):
        value = bundle[stat]
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
            raise ArtifactContractError(
                f"{source}: {stat} is {value!r}; the z-score transform for "
                f"{Z_SCORED_FEATURES[0]} cannot be applied.")


def load_artifacts(model_path: str | Path | None = None,
                   text_model_dir: str | Path | None = None,
                   require_text_model: bool = False) -> Artifacts:
    """Load and validate the trained artifacts. The only way in.

    Args:
        model_path: joblib bundle; defaults to ``RUBRIC_MODEL_PATH``.
        text_model_dir: DistilBERT directory; defaults to ``TEXT_MODEL_DIR``.
        require_text_model: raise if the text model is absent, instead of
            returning tabular-only artifacts.

    Raises:
        FileNotFoundError: the bundle (or a required text model) is not on disk.
        ArtifactContractError: it is on disk but does not match this contract.
    """
    # Imported after the existence check so "you have not trained a model yet"
    # is not reported as an unrelated ModuleNotFoundError on a bare checkout.
    model_path = Path(model_path) if model_path else RUBRIC_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No rubric model at {model_path}. Generate data and train first:\n"
            f"    python -m data.synthetic -n 200 --out data/synthetic_v1\n"
            f"    python -m scoring.train_xgboost --labels data/synthetic_v1/labels.csv "
            f"--plans data/synthetic_v1/plans --out {model_path}\n"
            f"(or set COTEACH_ARTIFACTS_DIR to a directory that has one)")

    import joblib

    bundle = joblib.load(model_path)
    validate_bundle(bundle, source=str(model_path))

    text_dir = Path(text_model_dir) if text_model_dir else TEXT_MODEL_DIR
    if not text_dir.is_dir():
        if require_text_model:
            raise FileNotFoundError(
                f"No DistilBERT text model at {text_dir}. It is not built yet "
                f"(scoring/train_bert_lora.py is a stub); call with "
                f"require_text_model=False for tabular-only explanations.")
        text_dir = None

    artifacts = Artifacts(bundle=bundle, model_path=model_path,
                          text_model_dir=text_dir)

    structural_fp, text_fp = artifacts.corpus_fingerprints
    if not artifacts.channels_consistent:
        logging.getLogger(__name__).warning(
            "Structural and text channels were trained on different corpora "
            "(%s vs %s). Their scores are not comparable and any fitted blend "
            "weights are invalid — retrain the stale channel and refit with "
            "evaluation/blend_weights.py. Blending is disabled until then.",
            structural_fp, text_fp)
    elif artifacts.provenance_unverified:
        logging.getLogger(__name__).warning(
            "Only one channel records its training corpus (structural=%s, "
            "text=%s), so they cannot be confirmed to share one. The channel "
            "without a fingerprint predates this check and may be stale — "
            "retrain it to remove the doubt.", structural_fp, text_fp)
    return artifacts


def corpus_fingerprint(labels_csv: str | Path) -> str:
    """Stable digest of a training corpus: plan ids and their rubric scores.

    The contract catches a feature-order mismatch, but nothing caught a
    *corpus* mismatch — and that bit. Regenerating the synthetic data while
    reusing a text checkpoint left the two channels fitted to different plans,
    and ``predict`` happily blended them into a score with no warning at all.
    Feature names matched, so every existing check passed.

    Recorded by both trainers and compared on load. Content-based, not a
    timestamp, so it survives copying artifacts between machines.
    """
    import csv
    import hashlib

    digest = hashlib.sha256()
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for record in sorted(csv.DictReader(f), key=lambda r: r["plan_id"]):
            row = [record["plan_id"]] + [record[k] for k in CRITERION_KEYS]
            digest.update("|".join(row).encode("utf-8"))
    return digest.hexdigest()[:16]


def order_features(features: dict[str, float]) -> list[float]:
    """Feature dict -> values in contract order, raising on any drift.

    Use this at every point a dict becomes a row, so a missing or stray key is
    caught here rather than becoming a misaligned SHAP attribution.
    """
    missing = [n for n in FEATURE_ORDER if n not in features]
    extra = [n for n in features if n not in FEATURE_ORDER]
    if missing or extra:
        raise ArtifactContractError(
            f"feature dict does not match contract {CONTRACT_VERSION}: "
            f"missing={missing}, unexpected={extra}")
    return [float(features[n]) for n in FEATURE_ORDER]


# --------------------------------------------------------------------------
# Import-time self-check
# --------------------------------------------------------------------------

def _self_check() -> None:
    """Fail at import if the repo has drifted away from this contract."""
    from ingestion.feature_engineer import FEATURE_NAMES

    live = tuple(FEATURE_NAMES)
    if live != FEATURE_ORDER:
        raise ArtifactContractError(
            "ingestion/feature_engineer.py FEATURE_NAMES has drifted from "
            f"model_contract.FEATURE_ORDER (contract {CONTRACT_VERSION}).\n"
            f"{_mismatch_report(FEATURE_ORDER, live)}\n"
            "Whichever changed, the trained models are now invalid: update both "
            "and retrain. See DEV.md, 'Feature order is a contract'.")

    if CRITERION_KEYS != tuple(RUBRIC_DIMENSIONS):
        raise ArtifactContractError(
            "rubric_schema.RUBRIC_DIMENSIONS has drifted from "
            f"model_contract.CRITERIA.\n{_mismatch_report(CRITERION_KEYS, tuple(RUBRIC_DIMENSIONS))}")

    duplicate_ids = len({c.id for c in CRITERIA}) != N_CRITERIA
    if duplicate_ids:
        raise ArtifactContractError("duplicate criterion IDs in CRITERIA")


_self_check()


if __name__ == "__main__":
    import json

    print(json.dumps({
        "contract_version": CONTRACT_VERSION,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "rubric_model": {"path": str(RUBRIC_MODEL_PATH),
                         "present": RUBRIC_MODEL_PATH.is_file()},
        "text_model": {"path": str(TEXT_MODEL_DIR),
                       "present": TEXT_MODEL_DIR.is_dir()},
        "n_features": N_FEATURES,
        "n_criteria": N_CRITERIA,
        "criteria": [
            {"id": c.id, "key": c.key, "label": c.label,
             "nts": c.nts_indicator, "nts_verified": c.nts_verified,
             "weight": c.weight}
            for c in CRITERIA
        ],
        "nts_source": NTS_SOURCE,
        "unverified_nts_codes": [c.id for c in unverified_criteria()],
    }, indent=2))
