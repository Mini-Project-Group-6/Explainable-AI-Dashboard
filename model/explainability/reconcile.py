"""Reconcile tabular SHAP values and text attributions into one explanation.

Per rubric criterion there are now two evidence channels:

    structural  XGBoost over 18 features, explained by SHAP TreeExplainer.
                Exact Shapley values, additive, in rubric score units:
                ``base_value + sum(shap_values) == the unrounded score``.
    text        DistilBERT+LoRA over the plan prose, explained by SHAP's
                Partition explainer over spans (``shap_deep.py``). Signed
                attributions in the text model's own output units.

The two are **not** on a common scale and are never summed. Adding a Shapley
value from one model to a Shapley value from a different model over a different
input space produces a number that decomposes nothing. What this module does
instead:

    scores    blended as a weighted mean of the two channels' predicted scores.
    evidence  each contribution normalised to its *share of total absolute
              attribution within its own channel* (0-1), then scaled by that
              channel's weight. That is a ranking quantity for display, not a
              claim that a feature and a sentence contributed the same amount.

Channel weights come from how far each channel can legitimately speak for the
criterion. Four criteria — resources/ICT, RPK introduction, attention to all
learners, closure — have no structural feature measuring them
(``feature_labels.uncovered_criteria()``), so their structural weight is near
zero and the explanation says so rather than dressing up a decomposition over
features that measure something else.

Deliberately dependency-free: plain floats and sequences in, dataclasses out.
It imports neither shap nor torch, so it can be unit-tested and reasoned about
without the model stack loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

from model_contract import (
    CRITERION_KEYS,
    FEATURE_ORDER,
    SCORE_MAX,
    SCORE_MIN,
    TEXT_MAX_TOKENS,
    Criterion,
    criterion,
)
from explainability.feature_labels import (
    FEATURE_LABELS,
    tabular_confidence,
    uncovered_criteria,
)

#: Weight the text channel gets before any measured reliability is available.
#: Provisional policy, not a fitted value: the text model reads the whole plan,
#: so it is given somewhat less than the structural channel where the structural
#: channel is strong, and everything where it is absent. Replace with per-
#: criterion held-out QWK once the real annotated CoE plans are scored (S5).
TEXT_CHANNEL_PRIOR: float = 0.60

#: base_value + sum(shap) should equal the model output to floating-point
#: slack. Beyond this the decomposition is not the one that produced the score.
ADDITIVITY_TOLERANCE: float = 1e-3

#: Below this ``tabular_confidence`` the structural features do not measure the
#: criterion at all (no primary feature maps to it), so their SHAP
#: decomposition is not evidence about it and is withheld. The floor sits
#: between the "indirect signal only" band (<=0.25) and the "has at least one
#: primary feature" band (>=0.75).
STRUCTURAL_EVIDENCE_FLOOR: float = 0.30


class Channel(str, Enum):
    STRUCTURAL = "structural"
    TEXT = "text"


@dataclass(frozen=True)
class SpanAttribution:
    """One attributed span of plan prose. Produced by ``shap_deep.py``.

    Declared here rather than there so the reconciler's input contract carries
    no dependency on shap/torch.
    """

    text: str
    value: float                # signed; positive raises the criterion score
    start: int = -1             # char offset into the plan's raw text
    end: int = -1
    section: str = ""           # canonical section name, if resolvable

    @property
    def quote(self) -> str:
        """Trimmed single-line form, for showing in a feedback panel."""
        collapsed = " ".join(self.text.split())
        return collapsed if len(collapsed) <= 160 else collapsed[:157] + "..."


@dataclass(frozen=True)
class Contribution:
    """One ranked piece of evidence, from either channel."""

    channel: Channel
    label: str                  # what the tutor reads
    detail: str                 # why it matters / the quoted text
    value: float                # signed, in the channel's own units
    influence: float            # |value| / total |values| in this channel, 0-1
    weight: float               # this channel's weight for this criterion
    rank_score: float           # influence * weight — the sort key
    feature: Optional[str] = None
    span: Optional[SpanAttribution] = None

    @property
    def raises_score(self) -> bool:
        return self.value > 0

    @property
    def direction(self) -> str:
        return "raises" if self.value > 0 else "lowers"

    def to_dict(self) -> dict:
        return {
            "channel": self.channel.value,
            "label": self.label,
            "detail": self.detail,
            "value": round(self.value, 4),
            "influence": round(self.influence, 4),
            "weight": round(self.weight, 4),
            "direction": self.direction,
            "feature": self.feature,
            "quote": self.span.quote if self.span else None,
            "section": self.span.section if self.span else None,
        }


@dataclass(frozen=True)
class CriterionExplanation:
    """The single per-criterion explanation the dashboard renders."""

    criterion: Criterion
    score: Optional[float]                   # blended, 1-4; None if no channel
    structural_score: Optional[float] = None
    text_score: Optional[float] = None
    structural_weight: float = 0.0
    text_weight: float = 0.0
    confidence: float = 0.0                  # 0-1, how well this criterion is evidenced
    evidence: tuple[Contribution, ...] = ()
    caveats: tuple[str, ...] = ()

    @property
    def rounded_score(self) -> Optional[int]:
        if self.score is None:
            return None
        return int(min(SCORE_MAX, max(SCORE_MIN, round(self.score))))

    @property
    def is_explainable(self) -> bool:
        """False when no channel can account for the score. Withhold the panel."""
        return bool(self.evidence) and self.confidence > 0.0

    def strengths(self, limit: int = 3) -> tuple[Contribution, ...]:
        return tuple(c for c in self.evidence if c.raises_score)[:limit]

    def weaknesses(self, limit: int = 3) -> tuple[Contribution, ...]:
        return tuple(c for c in self.evidence if not c.raises_score)[:limit]

    def to_dict(self) -> dict:
        return {
            "criterion_id": self.criterion.id,
            "criterion": self.criterion.key,
            "label": self.criterion.label,
            "nts_indicator": self.criterion.nts_indicator,
            "score": None if self.score is None else round(self.score, 3),
            "rounded_score": self.rounded_score,
            "channels": {
                "structural": {"score": self.structural_score,
                               "weight": round(self.structural_weight, 3)},
                "text": {"score": self.text_score,
                         "weight": round(self.text_weight, 3)},
            },
            "confidence": round(self.confidence, 3),
            "is_explainable": self.is_explainable,
            "evidence": [c.to_dict() for c in self.evidence],
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class PlanExplanation:
    """All ten criterion explanations for one plan, plus the overall score."""

    source: str
    criteria: tuple[CriterionExplanation, ...]
    overall_score_0_100: Optional[float] = None
    caveats: tuple[str, ...] = ()

    def by_key(self, key: str) -> CriterionExplanation:
        for item in self.criteria:
            if item.criterion.key == key:
                return item
        raise KeyError(key)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "overall_score_0_100": self.overall_score_0_100,
            "caveats": list(self.caveats),
            "criteria": [c.to_dict() for c in self.criteria],
        }


# --------------------------------------------------------------------------
# Channel weighting
# --------------------------------------------------------------------------

def channel_weights(criterion_key: str, *, structural_available: bool,
                    text_available: bool,
                    text_reliability: Optional[float] = None,
                    structural_reliability: Optional[float] = None,
                    structural_blend_weight: Optional[float] = None
                    ) -> tuple[float, float]:
    """(structural_weight, text_weight), summing to 1 when either is available.

    Three ways to decide the split, in descending order of trustworthiness:

        fitted        ``structural_blend_weight`` — the weight that minimises
                      held-out error, from evaluation/blend_weights.py. Used
                      whenever it is available; everything below is a fallback.
        measured      both channels' held-out agreement, compared like for like.
        heuristic     coverage vs a flat prior, when nothing is measured.

    Two quantities are involved in the fallbacks and they must not be confused:

        coverage      ``tabular_confidence`` — does any feature *measure* this
                      criterion? Gates whether the structural channel may offer
                      evidence at all (``STRUCTURAL_EVIDENCE_FLOOR``).
        reliability   measured held-out agreement — how *accurate* is a channel
                      on this criterion? Decides how much its score is worth.

    An earlier version weighted the structural side by coverage and the text
    side by measured QWK. Those are different kinds of number, and the result
    was perverse: the text model measurably got worse on every criterion while
    its weight went *up*, because a QWK of 0.72 outranks... nothing comparable.

    So weights are only computed from measured agreement when **both** sides
    are measured. Otherwise both fall back to their heuristics — a like-for-like
    comparison, even if a cruder one. Never mix the two.
    """
    if not structural_available and not text_available:
        return 0.0, 0.0
    if not structural_available:
        return 0.0, 1.0
    if not text_available:
        return 1.0, 0.0

    # Best available: a weight fitted to minimise held-out error on plans both
    # channels were validated against (evaluation/blend_weights.py). Preferred
    # over any accuracy ratio, which answers a question we were not asking.
    if structural_blend_weight is not None:
        w = max(0.0, min(1.0, structural_blend_weight))
        return w, 1.0 - w

    both_measured = (structural_reliability is not None
                     and text_reliability is not None)
    structural = (max(0.0, min(1.0, structural_reliability)) if both_measured
                  else tabular_confidence(criterion_key))
    text = (max(0.0, min(1.0, text_reliability)) if both_measured
            else TEXT_CHANNEL_PRIOR)

    total = structural + text
    if total <= 0.0:
        return 0.0, 0.0
    return structural / total, text / total


# --------------------------------------------------------------------------
# Per-channel contribution building
# --------------------------------------------------------------------------

def _structural_contributions(shap_values: Sequence[float],
                              feature_values: Optional[Mapping[str, float]],
                              weight: float) -> list[Contribution]:
    if len(shap_values) != len(FEATURE_ORDER):
        raise ValueError(
            f"expected {len(FEATURE_ORDER)} SHAP values (one per contract "
            f"feature), got {len(shap_values)}. SHAP columns are positional — "
            f"a length mismatch means they are misaligned.")

    magnitude = sum(abs(float(v)) for v in shap_values)
    contributions = []
    for name, value in zip(FEATURE_ORDER, shap_values):
        value = float(value)
        label = FEATURE_LABELS[name]
        detail = label.meaning
        if feature_values is not None and name in feature_values:
            detail = f"{detail} This plan: {_format_value(feature_values[name], label.unit)}."
        influence = abs(value) / magnitude if magnitude else 0.0
        contributions.append(Contribution(
            channel=Channel.STRUCTURAL,
            label=label.label,
            detail=detail,
            value=value,
            influence=influence,
            weight=weight,
            rank_score=influence * weight,
            feature=name,
        ))
    return contributions


def _text_contributions(spans: Iterable[SpanAttribution],
                        weight: float) -> list[Contribution]:
    spans = [s for s in spans if s.text.strip()]
    magnitude = sum(abs(s.value) for s in spans)
    contributions = []
    for span in spans:
        influence = abs(span.value) / magnitude if magnitude else 0.0
        where = f"From the {span.section} section. " if span.section else ""
        contributions.append(Contribution(
            channel=Channel.TEXT,
            label=f'"{span.quote}"',
            detail=f"{where}Wording the text model weighted for this criterion.",
            value=span.value,
            influence=influence,
            weight=weight,
            rank_score=influence * weight,
            span=span,
        ))
    return contributions


def _format_value(value: float, unit: str) -> str:
    if unit.startswith("proportion") or unit.startswith("similarity") \
            or unit.startswith("agreement"):
        return f"{float(value):.0%}"
    if float(value).is_integer():
        return f"{int(value)} {unit.split(' (')[0]}"
    return f"{float(value):.2f} {unit.split(' (')[0]}"


def _interleave(contributions: list[Contribution], top_k: int,
                active_channels: set[Channel]) -> tuple[Contribution, ...]:
    """Top-k by rank score, but never silently drop an entire active channel."""
    ranked = sorted(contributions, key=lambda c: c.rank_score, reverse=True)
    chosen = [c for c in ranked[:top_k]]
    represented = {c.channel for c in chosen}
    for channel in active_channels - represented:
        best = next((c for c in ranked if c.channel == channel), None)
        if best is not None:
            if len(chosen) >= top_k and chosen:
                chosen.pop()
            chosen.append(best)
    return tuple(sorted(chosen, key=lambda c: c.rank_score, reverse=True))


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

def reconcile_criterion(
    criterion_key: str,
    *,
    structural_score: Optional[float] = None,
    shap_values: Optional[Sequence[float]] = None,
    shap_base_value: Optional[float] = None,
    feature_values: Optional[Mapping[str, float]] = None,
    text_score: Optional[float] = None,
    text_spans: Optional[Sequence[SpanAttribution]] = None,
    text_reliability: Optional[float] = None,
    structural_reliability: Optional[float] = None,
    structural_blend_weight: Optional[float] = None,
    text_token_count: Optional[int] = None,
    top_k: int = 6,
) -> CriterionExplanation:
    """Fold both channels into one explanation for a single criterion.

    Every channel argument is optional; whichever is supplied is used. With
    neither, the result carries ``score=None`` and a caveat rather than a
    fabricated number.
    """
    target = criterion(criterion_key)
    structural_available = structural_score is not None and shap_values is not None
    text_available = text_score is not None

    # A channel that cannot produce evidence for this criterion does not vote on
    # its score either — unless it is the only channel there is, in which case
    # the score is passed through and marked unexplained. Showing a number
    # partly driven by a decomposition the tutor is not allowed to see would
    # defeat the point of the tool.
    structural_explains = (
        structural_available
        and tabular_confidence(criterion_key) >= STRUCTURAL_EVIDENCE_FLOOR)

    w_structural, w_text = channel_weights(
        criterion_key,
        structural_available=structural_explains,
        text_available=text_available,
        text_reliability=text_reliability,
        structural_reliability=structural_reliability,
        structural_blend_weight=structural_blend_weight)

    caveats: list[str] = []
    contributions: list[Contribution] = []
    active: set[Channel] = set()

    if structural_available:
        if structural_explains and w_structural > 0:
            contributions += _structural_contributions(
                shap_values, feature_values, w_structural)
            active.add(Channel.STRUCTURAL)
        if shap_base_value is not None:
            reconstructed = float(shap_base_value) + sum(float(v) for v in shap_values)
            if abs(reconstructed - float(structural_score)) > ADDITIVITY_TOLERANCE:
                caveats.append(
                    f"SHAP additivity check failed for the structural channel: "
                    f"base + contributions = {reconstructed:.4f} but the model "
                    f"predicted {float(structural_score):.4f}. The breakdown may "
                    f"not correspond to this score.")

    if text_available and text_spans:
        contributions += _text_contributions(text_spans, w_text)
        if w_text > 0:
            active.add(Channel.TEXT)

    # --- blended score ---
    score: Optional[float] = None
    if structural_available or text_available:
        total_weight = (w_structural if structural_explains else 0.0) + \
                       (w_text if text_available else 0.0)
        if total_weight > 0:
            score = ((w_structural * float(structural_score) if structural_explains else 0.0)
                     + (w_text * float(text_score) if text_available else 0.0)) / total_weight
        else:
            # Both weights zero (uncovered criterion, no text model): pass the
            # only score there is through, and say it is unexplained.
            score = float(structural_score) if structural_available else float(text_score)
        score = min(float(SCORE_MAX), max(float(SCORE_MIN), score))

    structural_quality = (tabular_confidence(criterion_key)
                          if structural_reliability is None
                          else structural_reliability)
    text_quality = TEXT_CHANNEL_PRIOR if text_reliability is None else text_reliability
    confidence = (w_structural * structural_quality
                  + w_text * text_quality) if active else 0.0

    # --- caveats ---
    if criterion_key in uncovered_criteria():
        if text_available:
            caveats.append(
                f"No structural feature measures {target.label.lower()}; this "
                f"score and its evidence come from the plan's wording.")
        else:
            caveats.append(
                f"No structural feature measures {target.label.lower()}, and the "
                f"text model is not loaded. The score is shown without an "
                f"explanation — do not present the feature breakdown as evidence "
                f"for this criterion.")
    elif structural_available and not text_available and \
            tabular_confidence(criterion_key) < 0.75:
        caveats.append(
            f"{target.label} is only indirectly measured by the structural "
            f"features; treat the breakdown as indicative.")

    if not text_available:
        caveats.append("Text model not loaded — structural channel only.")
    elif text_token_count is not None and text_token_count > TEXT_MAX_TOKENS:
        caveats.append(
            f"The text model read only the first {TEXT_MAX_TOKENS} of "
            f"{text_token_count} tokens; wording after that is not attributed.")

    if not active:
        caveats.append("No evidence channel is available for this criterion.")

    return CriterionExplanation(
        criterion=target,
        score=score,
        structural_score=None if structural_score is None else float(structural_score),
        text_score=None if text_score is None else float(text_score),
        structural_weight=w_structural,
        text_weight=w_text,
        confidence=confidence,
        evidence=_interleave(contributions, top_k, active),
        caveats=tuple(caveats),
    )


def reconcile_plan(
    *,
    source: str = "<plan>",
    structural_scores: Optional[Mapping[str, float]] = None,
    shap_values: Optional[Mapping[str, Sequence[float]]] = None,
    shap_base_values: Optional[Mapping[str, float]] = None,
    feature_values: Optional[Mapping[str, float]] = None,
    text_scores: Optional[Mapping[str, float]] = None,
    text_spans: Optional[Mapping[str, Sequence[SpanAttribution]]] = None,
    text_reliability: Optional[Mapping[str, float]] = None,
    structural_reliability: Optional[Mapping[str, float]] = None,
    structural_blend_weight: Optional[Mapping[str, float]] = None,
    text_token_count: Optional[int] = None,
    top_k: int = 6,
) -> PlanExplanation:
    """Reconcile all ten criteria for one plan.

    Mappings are keyed by criterion key; missing entries mean "that channel has
    nothing for this criterion", which is handled rather than raised on.
    """
    explanations = []
    for key in CRITERION_KEYS:
        explanations.append(reconcile_criterion(
            key,
            structural_score=(structural_scores or {}).get(key),
            shap_values=(shap_values or {}).get(key),
            shap_base_value=(shap_base_values or {}).get(key),
            feature_values=feature_values,
            text_score=(text_scores or {}).get(key),
            text_spans=(text_spans or {}).get(key),
            text_reliability=(text_reliability or {}).get(key),
            structural_reliability=(structural_reliability or {}).get(key),
            structural_blend_weight=(structural_blend_weight or {}).get(key),
            text_token_count=text_token_count,
            top_k=top_k,
        ))

    overall = None
    scored = {e.criterion.key: e.rounded_score for e in explanations
              if e.rounded_score is not None}
    if len(scored) == len(CRITERION_KEYS):
        from rubric_schema import overall_score
        overall = round(overall_score(scored), 2)

    plan_caveats = []
    unexplained = [e.criterion.id for e in explanations if not e.is_explainable]
    if unexplained:
        plan_caveats.append(
            f"{len(unexplained)} of {len(CRITERION_KEYS)} criteria "
            f"({', '.join(unexplained)}) have no usable evidence channel.")

    return PlanExplanation(source=source, criteria=tuple(explanations),
                           overall_score_0_100=overall,
                           caveats=tuple(plan_caveats))
