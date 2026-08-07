"""XAI UI components — turning a reconciled explanation into something readable.

Two layers, deliberately separated:

    view models   ``criterion_view`` / ``plan_view`` produce plain dicts. No
                  framework, no imports beyond stdlib, unit-testable.
    components    ``st_*`` functions render those into Streamlit. Streamlit is
                  imported lazily inside each one, so S3's dashboard can depend
                  on this module without the model layer depending on Streamlit.

Accessibility rules applied throughout, because the finding this project
reports is about *trust* and a panel a tutor cannot read is not evidence:

  * colour is never the only signal — every contribution also carries a "+"/"-"
    and the words "raises"/"lowers";
  * the diverging pair is blue (#2166AC, raises) and red (#B2182B, lowers),
    which stays distinguishable under the common colour-vision deficiencies,
    rather than the red/green default;
  * bar length encodes influence share within a channel, and the channel is
    named on every row, so a SHAP value and a text attribution are never shown
    as if they were the same measurement.

Where a criterion has no trustworthy evidence channel (see
``feature_labels.uncovered_criteria``) the components render an explicit
"not explained" state instead of a chart. That is the honest rendering, and it
is what stops the dashboard from implying evidence it does not have.
"""

from __future__ import annotations

import html
from typing import Iterable, Optional, Sequence

from model_contract import CRITERIA, N_CRITERIA, Artifacts, criterion
from explainability.feature_labels import (
    FEATURE_LABELS,
    coverage_report,
    tabular_confidence,
    uncovered_criteria,
)
from explainability.reconcile import (
    Channel,
    Contribution,
    CriterionExplanation,
    PlanExplanation,
    SpanAttribution,
)
from explainability.suggestions import Suggestion

POSITIVE_COLOUR = "#2166AC"   # raises the score
NEGATIVE_COLOUR = "#B2182B"   # lowers the score
NEUTRAL_COLOUR = "#6B7280"

BAND_LABELS: tuple[tuple[float, str], ...] = (
    (85.0, "Outstanding"),
    (70.0, "Good"),
    (50.0, "Minimum level of practice"),
    (0.0, "Inadequate"),
)


def band_for(score_0_100: Optional[float]) -> str:
    """Rubric band name for an overall 0-100 score."""
    if score_0_100 is None:
        return "Not scored"
    for threshold, name in BAND_LABELS:
        if score_0_100 >= threshold:
            return name
    return "Inadequate"


# --------------------------------------------------------------------------
# View models
# --------------------------------------------------------------------------

def contribution_view(contribution: Contribution) -> dict:
    """One evidence row, ready to render."""
    return {
        "channel": contribution.channel.value,
        "channel_label": ("Plan structure" if contribution.channel is Channel.STRUCTURAL
                          else "Plan wording"),
        "label": contribution.label,
        "detail": contribution.detail,
        "direction": contribution.direction,
        "sign": "+" if contribution.raises_score else "-",
        "colour": POSITIVE_COLOUR if contribution.raises_score else NEGATIVE_COLOUR,
        "bar_pct": round(min(1.0, contribution.influence) * 100, 1),
        "influence": round(contribution.influence, 4),
        "value": round(contribution.value, 4),
        "feature": contribution.feature,
        "quote": contribution.span.quote if contribution.span else None,
        "section": contribution.span.section if contribution.span else None,
    }


def criterion_view(explanation: CriterionExplanation) -> dict:
    """Everything one criterion card needs."""
    target = explanation.criterion
    return {
        "id": target.id,
        "key": target.key,
        "label": target.label,
        "nts_indicator": target.nts_indicator,
        "nts_descriptor": target.nts_descriptor,
        "nts_verified": target.nts_verified,
        "plan_evidence": target.plan_evidence,
        "weight": target.weight,
        "score": explanation.rounded_score,
        "score_exact": (None if explanation.score is None
                        else round(explanation.score, 2)),
        "confidence": round(explanation.confidence, 2),
        "confidence_label": _confidence_label(explanation.confidence),
        "is_explainable": explanation.is_explainable,
        "channels": {
            "structural": {"score": explanation.structural_score,
                           "weight": round(explanation.structural_weight, 2)},
            "text": {"score": explanation.text_score,
                     "weight": round(explanation.text_weight, 2)},
        },
        "strengths": [contribution_view(c) for c in explanation.strengths()],
        "weaknesses": [contribution_view(c) for c in explanation.weaknesses()],
        "evidence": [contribution_view(c) for c in explanation.evidence],
        "caveats": list(explanation.caveats),
    }


def plan_view(explanation: PlanExplanation,
              suggestions: Sequence[Suggestion] = ()) -> dict:
    """The whole submission, ready for the scoring tab."""
    criteria = [criterion_view(item) for item in explanation.criteria]
    explained = sum(1 for item in criteria if item["is_explainable"])
    return {
        "source": explanation.source,
        "overall_score_0_100": explanation.overall_score_0_100,
        "band": band_for(explanation.overall_score_0_100),
        "criteria": criteria,
        "explained_count": explained,
        "criteria_count": N_CRITERIA,
        "caveats": list(explanation.caveats),
        "suggestions": [s.to_dict() for s in suggestions],
    }


def _confidence_label(confidence: float) -> str:
    if confidence <= 0.0:
        return "Not explained"
    if confidence < 0.4:
        return "Indicative only"
    if confidence < 0.75:
        return "Moderate"
    return "Well evidenced"


# --------------------------------------------------------------------------
# HTML fragments (framework-free, testable)
# --------------------------------------------------------------------------

def contribution_bars_html(contributions: Iterable[Contribution],
                           max_rows: int = 6) -> str:
    """Diverging influence bars for one criterion's evidence."""
    rows = []
    for contribution in list(contributions)[:max_rows]:
        view = contribution_view(contribution)
        rows.append(
            '<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="flex:0 0 46%;font-size:0.86rem;">'
            f'{html.escape(str(view["label"]))}'
            f'<span style="color:{NEUTRAL_COLOUR};font-size:0.76rem;"> &middot; '
            f'{html.escape(view["channel_label"])}</span></div>'
            f'<div style="flex:0 0 3.2rem;text-align:right;font-variant-numeric:'
            f'tabular-nums;color:{view["colour"]};font-weight:600;">'
            f'{view["sign"]}{view["bar_pct"]:.0f}%</div>'
            '<div style="flex:1;background:rgba(128,128,128,0.15);height:10px;'
            'border-radius:5px;overflow:hidden;">'
            f'<div style="width:{view["bar_pct"]}%;background:{view["colour"]};'
            'height:100%;"></div></div>'
            f'<div style="flex:0 0 4.2rem;font-size:0.76rem;color:{NEUTRAL_COLOUR};">'
            f'{html.escape(view["direction"])}</div>'
            '</div>')
    if not rows:
        return (f'<p style="color:{NEUTRAL_COLOUR};font-style:italic;">'
                'No evidence available for this criterion.</p>')
    return "".join(rows)


def highlight_spans_html(raw_text: str, spans: Sequence[SpanAttribution],
                         max_spans: int = 12) -> str:
    """The plan text with its most influential spans highlighted.

    Intensity encodes absolute attribution; hue encodes direction. Overlapping
    spans are resolved in favour of the stronger one, and spans that could not
    be located (offset -1) are simply not highlighted rather than guessed at.
    """
    ranked = sorted((s for s in spans if s.start >= 0 and s.end > s.start),
                    key=lambda s: abs(s.value), reverse=True)[:max_spans]
    if not ranked:
        return (f'<pre style="white-space:pre-wrap;font-family:inherit;">'
                f'{html.escape(raw_text)}</pre>')

    peak = max(abs(s.value) for s in ranked) or 1.0

    chosen: list[SpanAttribution] = []
    for span in ranked:                      # strongest first, so it wins overlaps
        if any(span.start < other.end and other.start < span.end for other in chosen):
            continue
        chosen.append(span)
    chosen.sort(key=lambda s: s.start)

    parts, cursor = [], 0
    for span in chosen:
        start, end = max(span.start, cursor), min(span.end, len(raw_text))
        if start >= end:
            continue
        parts.append(html.escape(raw_text[cursor:start]))
        colour = POSITIVE_COLOUR if span.value > 0 else NEGATIVE_COLOUR
        alpha = 0.15 + 0.45 * (abs(span.value) / peak)
        sign = "+" if span.value > 0 else "-"
        tooltip = (f"{sign}{abs(span.value):.3f} "
                   f"({'raises' if span.value > 0 else 'lowers'} the score)"
                   + (f" - {span.section} section" if span.section else ""))
        parts.append(
            f'<mark title="{html.escape(tooltip)}" '
            f'style="background-color:{_rgba(colour, alpha)};'
            f'border-bottom:2px solid {colour};padding:0 1px;">'
            f'{html.escape(raw_text[start:end])}</mark>')
        cursor = end
    parts.append(html.escape(raw_text[cursor:]))

    return ('<pre style="white-space:pre-wrap;font-family:inherit;line-height:1.6;">'
            + "".join(parts) + "</pre>")


def _rgba(hex_colour: str, alpha: float) -> str:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def legend_html() -> str:
    """Colour key. Shown once per tab so the encoding is never unexplained."""
    return (
        '<div style="display:flex;gap:16px;align-items:center;font-size:0.8rem;'
        f'color:{NEUTRAL_COLOUR};margin:6px 0 12px;">'
        f'<span><span style="display:inline-block;width:11px;height:11px;'
        f'background:{POSITIVE_COLOUR};border-radius:2px;"></span> '
        '&nbsp;+ raises the score</span>'
        f'<span><span style="display:inline-block;width:11px;height:11px;'
        f'background:{NEGATIVE_COLOUR};border-radius:2px;"></span> '
        '&nbsp;&minus; lowers the score</span>'
        '<span>Bar length = share of that channel&rsquo;s total influence</span>'
        '</div>')


# --------------------------------------------------------------------------
# Streamlit components
# --------------------------------------------------------------------------

def st_overall_summary(explanation: PlanExplanation) -> None:
    """Headline score, band, and how much of it is actually explained."""
    import streamlit as st

    view = plan_view(explanation)
    left, middle, right = st.columns(3)
    score = view["overall_score_0_100"]
    left.metric("Overall score", "-" if score is None else f"{score:.1f} / 100")
    middle.metric("Band", view["band"])
    right.metric("Criteria explained",
                 f"{view['explained_count']} / {view['criteria_count']}")

    for caveat in view["caveats"]:
        st.warning(caveat, icon=":material/info:")


def st_criterion_card(explanation: CriterionExplanation,
                      show_legend: bool = False) -> None:
    """One criterion: score, NTS tag, evidence bars, caveats."""
    import streamlit as st

    view = criterion_view(explanation)
    with st.expander(
            f"{view['id']}  {view['label']}  "
            f"({'-' if view['score'] is None else view['score']}/4)",
            expanded=False):
        st.caption(
            f"NTS {view['nts_indicator']} - {view['nts_descriptor']}"
            + ("" if view["nts_verified"] else "  *(indicator code pending "
                                               "verification against the handbook)*"))
        st.caption(f"What a tutor looks for: {view['plan_evidence']}")

        if show_legend:
            st.markdown(legend_html(), unsafe_allow_html=True)

        if not view["is_explainable"]:
            st.info(
                "This criterion is scored but not explained — no evidence "
                "channel measures it. See the caveats below.",
                icon=":material/help:")
        else:
            st.markdown(f"**Confidence: {view['confidence_label']}**")
            st.markdown(contribution_bars_html(explanation.evidence),
                        unsafe_allow_html=True)

        for caveat in view["caveats"]:
            st.caption(f":grey[{caveat}]")


def st_criteria_panel(explanation: PlanExplanation) -> None:
    """All ten criterion cards, rubric order."""
    import streamlit as st

    st.markdown(legend_html(), unsafe_allow_html=True)
    for item in explanation.criteria:
        st_criterion_card(item, show_legend=False)


def st_highlighted_plan(raw_text: str, spans: Sequence[SpanAttribution],
                        title: str = "Where this came from in your plan") -> None:
    """The text channel's attribution, highlighted over the plan itself."""
    import streamlit as st

    st.subheader(title)
    if not spans:
        st.caption("The text model is not loaded, so no wording is highlighted.")
        return
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.markdown(highlight_spans_html(raw_text, spans), unsafe_allow_html=True)


def st_suggestions_panel(suggestions: Sequence[Suggestion]) -> None:
    """The revision list — the part a student teacher actually acts on.

    Objective 22 requires a feature-importance explanation on every feedback
    item, so each suggestion shows the Shapley value that triggered it (or
    states plainly that it came from a content check instead).
    """
    import streamlit as st

    st.subheader("Suggested revisions")
    if not suggestions:
        st.success("No revision suggestions were triggered for this plan.")
        return

    icons = {"high": ":material/priority_high:", "medium": ":material/chevron_right:",
             "low": ":material/remove:"}
    for suggestion in suggestions:
        with st.container(border=True):
            st.markdown(
                f"{icons.get(suggestion.priority, '')} **{suggestion.message}**")
            st.caption(
                f"{suggestion.criterion_id} {suggestion.criterion_label} "
                f"(NTS {suggestion.nts_indicator}) &middot; {suggestion.evidence}")
            with st.expander("Why the model flagged this", expanded=False):
                st.markdown(suggestion.attribution_text)
                if suggestion.has_shap_explanation:
                    st.markdown(
                        contribution_bars_html([_as_contribution(suggestion)]),
                        unsafe_allow_html=True)


def _as_contribution(suggestion: Suggestion) -> Contribution:
    """Render a suggestion's SHAP basis with the same bar as the evidence rows."""
    return Contribution(
        channel=Channel.STRUCTURAL,
        label=suggestion.feature_label or suggestion.feature or "",
        detail=suggestion.evidence,
        value=suggestion.shap_value or 0.0,
        influence=suggestion.influence or 0.0,
        weight=1.0,
        rank_score=suggestion.influence or 0.0,
        feature=suggestion.feature,
    )


# --- SHAP-native plots -----------------------------------------------------
# S2 owns these so S3 never has to import shap or manage a matplotlib figure.

def st_waterfall(explanation, criterion_label: str = "") -> None:
    """SHAP waterfall for one plan x one criterion (objective 23)."""
    import matplotlib.pyplot as plt
    import streamlit as st

    from explainability.shap_tree import _with_readable_labels

    import shap

    shap.plots.waterfall(_with_readable_labels(explanation), max_display=12,
                         show=False)
    figure = plt.gcf()
    if criterion_label:
        figure.suptitle(criterion_label)
    figure.set_size_inches(9, 6)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def st_force_plot(explanation, height: int = 180) -> None:
    """Interactive SHAP force plot, embedded as a component."""
    import streamlit as st
    import streamlit.components.v1 as components

    import shap

    from explainability.shap_tree import force_plot_html

    visualiser = force_plot_html(explanation)
    components.html(f"{shap.getjs()}{visualiser.html()}", height=height)


def st_beeswarm(explanation, criterion_label: str = "") -> None:
    """Corpus-level beeswarm summary — the cohort view for tutors."""
    import matplotlib.pyplot as plt
    import streamlit as st

    import shap

    from explainability.shap_tree import _with_readable_labels

    shap.plots.beeswarm(_with_readable_labels(explanation), max_display=18,
                        show=False)
    figure = plt.gcf()
    if criterion_label:
        figure.suptitle(criterion_label)
    figure.set_size_inches(9, 7)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def st_transparency_panel(artifacts: Optional[Artifacts] = None) -> None:
    """The transparency tab: what the model is, and what it cannot see.

    Stating the coverage gap in the UI is the point. A tutor who knows four
    criteria are scored from prose alone can calibrate their trust; one who is
    shown ten identical-looking panels cannot.
    """
    import streamlit as st

    from model_contract import (
        CONTRACT_VERSION,
        NTS_SOURCE,
        N_FEATURES,
        unverified_criteria,
    )

    st.subheader("How this model works")
    st.markdown(
        f"- **Contract version:** `{CONTRACT_VERSION}`\n"
        f"- **Structural features:** {N_FEATURES}, extracted deterministically "
        f"from the uploaded plan\n"
        f"- **Rubric:** {N_CRITERIA} plan-assessable criteria. The STS lesson "
        f"observation checklist has 25 items; the other 15 assess live teaching "
        f"delivery and cannot be judged from a written plan.\n"
        f"- **Text channel:** "
        + ("loaded" if artifacts and artifacts.has_text_model else "not loaded"))

    st.subheader("What the structural features can and cannot measure")
    report = coverage_report()
    st.dataframe(
        [
            {
                "Criterion": f"{item.id} {item.label}",
                "NTS": item.nts_indicator,
                "Direct features": ", ".join(
                    FEATURE_LABELS[n].code for n in report[item.key]["primary"]) or "none",
                "Indirect": ", ".join(
                    FEATURE_LABELS[n].code for n in report[item.key]["secondary"]) or "-",
                "Structural confidence": f"{tabular_confidence(item.key):.2f}",
            }
            for item in CRITERIA
        ],
        hide_index=True, use_container_width=True)

    gaps = uncovered_criteria()
    if gaps:
        names = ", ".join(criterion(k).label for k in gaps)
        st.warning(
            f"No structural feature directly measures: {names}. These are "
            f"scored from the plan's wording where the text model is available, "
            f"and are shown as unexplained where it is not.",
            icon=":material/warning:")

    pending = unverified_criteria()
    if pending:
        st.caption(
            f":grey[NTS indicator codes are matched by descriptor wording against "
            f"{NTS_SOURCE} and are pending confirmation of their letter suffixes "
            f"for {len(pending)} of {N_CRITERIA} criteria.]")

    st.subheader("Feature reference")
    st.dataframe(
        [
            {
                "Code": label.code,
                "Shown as": label.label,
                "Means": label.meaning,
                "Better when": label.direction.value.replace("_", " "),
                "Criterion": label.criterion.id,
            }
            for label in FEATURE_LABELS.values()
        ],
        hide_index=True, use_container_width=True)
