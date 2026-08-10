# pages/dashboard.py
"""The dashboard shell. S3 owns the page; S2 owns every explanation panel.

The first draft rendered a rubric of its own: six invented dimensions
("Differentiation", "Resource Integration", ...) scored 0-100, with hand-drawn
SHAP waterfall and force plots over hardcoded numbers. None of it matched what
S1 trained or what S2 renders — the live rubric is ten NTS-tagged criteria
scored 1-4, and only the weighted overall is 0-100.

So nothing on this page describes the model any more. It lays out the four tabs
from DEV.md, handles the upload, and calls into ``explainability.render``:

    st_overall_summary      headline score, band, how much is explained
    st_criteria_panel       the ten criterion cards with evidence bars
    st_suggestions_panel    revisions, each carrying its SHAP attribution
    st_waterfall/st_force_plot   the SHAP plots, drawn by S2 from real values
    st_transparency_panel   what the model is and what it cannot see

Where the model layer is not importable this page says so plainly instead of
substituting placeholder scores. A dashboard that shows plausible invented
numbers is the exact failure mode this project is measuring.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import accounts, session
from app.components.sidebar import dashboard_sidebar, muted
from app.components.title import page_title
from app.database import db
from app.model_bridge import artifacts_status, model_layer
from app.scoring import (
    ACCEPTED_TYPES,
    get_artifacts,
    score_upload,
    scorer_error,
    shap_explanations,
    submission_id,
)

page_title(layout="wide")
user = session.require_user()
dashboard_sidebar(user)

GREY = muted()
BRAND = "#1D4ED8"

st.markdown(
    f"""
    <style>
    .block-container {{ max-width: 1240px; padding: 2.4rem 2rem 4rem; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid rgba(128,128,128,0.25); }}
    /* Streamlit's auto page nav would list "main" beside "dashboard" */
    [data-testid="stSidebarNav"] {{ display: none; }}

    /* Drop zone. Tinted rather than a fixed near-white, which was invisible
       against the page in light mode and glaring in dark mode. */
    [data-testid="stFileUploader"] section {{
        background: rgba(148,163,184,0.09);
        border: 2px dashed rgba(148,163,184,0.45);
        border-radius: 14px;
        padding: 22px;
    }}
    [data-testid="stFileUploader"] button {{
        background:{BRAND}; color:#fff; border:none; border-radius:10px;
    }}
    [data-testid="stFileUploader"] button:hover {{ background:#2563EB; color:#fff; }}
    .page-sub {{ color:{GREY}; margin-top:2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
head_left, head_right = st.columns([3, 1.4], vertical_alignment="center")
with head_left:
    st.markdown(
        "<h1 style='margin-bottom:0;'>Explainable AI Co-Teaching Dashboard</h1>"
        "<p class='page-sub'>Teacher College Research Portal · "
        "Academic Year 2025&ndash;2026</p>",
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;justify-content:flex-end;">
            <div style="text-align:right;min-width:0;">
                <div style="font-weight:700;overflow:hidden;
                            text-overflow:ellipsis;">{user.name}</div>
                <div style="font-size:13px;color:{GREY};overflow:hidden;
                            text-overflow:ellipsis;">{user.email}</div>
            </div>
            <div style="flex:0 0 auto;background:{BRAND};color:#fff;width:44px;
                        height:44px;border-radius:50%;display:flex;
                        align-items:center;justify-content:center;
                        font-weight:700;">{user.initials}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

layer = model_layer()
if not layer.available:
    st.error(
        f"**The scoring model is not loaded, so no plan can be scored.**\n\n"
        f"{layer.error}\n\n{layer.hint}",
        icon=":material/error:",
    )

scoring_tab, history_tab, survey_tab, transparency_tab = st.tabs(
    ["Scoring", "Revision history", "Trust survey", "Transparency"])


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
with scoring_tab:
    with st.container(border=True):
        st.markdown("#### Lesson plan upload")
        upload = st.file_uploader(
            "Upload a lesson plan",
            type=ACCEPTED_TYPES,
            label_visibility="collapsed",
            disabled=not layer.available,
            help="PDF or Word lesson plan. Plain text is accepted for testing.",
        )
        if layer.available and scorer_error():
            st.warning(scorer_error(), icon=":material/warning:")

    submission = None
    if upload is not None and layer.available and scorer_error() is None:
        with st.spinner("Scoring the plan and building its explanation…"):
            try:
                submission = score_upload(upload)
            except Exception as error:                # noqa: BLE001
                st.error(f"Could not score this plan — "
                         f"{type(error).__name__}: {error}",
                         icon=":material/error:")

    if submission is not None:
        # Streamlit re-runs this script on every interaction, so the write has
        # to be guarded on the submission's identity. Without it, each tab
        # click appended another revision-history row for the same plan.
        recorded = st.session_state.setdefault("recorded_submissions", set())
        key = submission_id(upload)
        if key not in recorded:
            recorded.add(key)
            saved = db.record_submission(
                user_email=user.email,
                source=submission["view"]["source"],
                overall_score=submission["overall_score_0_100"],
                band=submission["view"]["band"],
                rubric_scores=submission["rubric_scores"],
                contract_version=getattr(layer.contract, "CONTRACT_VERSION", None),
            )
            if not saved:
                st.session_state.setdefault("session_history", []).append({
                    "source": submission["view"]["source"],
                    "overall_score": submission["overall_score_0_100"],
                    "band": submission["view"]["band"],
                })

    if submission is None:
        st.caption("Upload a lesson plan to see its rubric scores and the "
                   "evidence behind them.")
    else:
        render = layer.render
        explanation = submission["explanation"]

        st.write("")
        with st.container(border=True):
            render.st_overall_summary(explanation)
            meta = [f"`{submission['view']['source']}`"]
            if submission["latency_seconds"] is not None:
                meta.append(f"scored in {submission['latency_seconds']:.1f}s")
            meta.append("plan structure + wording" if submission["used_text_channel"]
                        else "plan structure only")
            st.caption(" · ".join(meta))
            if submission["missing_sections"]:
                st.caption(":grey[Sections not found in the document: "
                           + ", ".join(submission["missing_sections"]) + "]")

        st.write("")
        with st.container(border=True):
            st.markdown("#### Rubric criteria")
            st.caption("Each criterion is scored 1–4 against its National "
                       "Teachers' Standards indicator. Open a card to see the "
                       "evidence behind the score.")
            render.st_criteria_panel(explanation)

        st.write("")
        with st.container(border=True):
            st.markdown("#### SHAP explanation")
            explainable = [item for item in explanation.criteria
                           if item.is_explainable]
            if not explainable:
                st.info("No criterion on this plan has a trustworthy evidence "
                        "channel, so there is nothing to decompose.",
                        icon=":material/help:")
            else:
                labels = {f"{item.criterion.id}  {item.criterion.label}": item
                          for item in explainable}
                chosen = st.selectbox("Criterion", list(labels),
                                      key="shap_criterion")
                st.caption(
                    "Values are in rubric score units (1–4), not points out of "
                    "100. The plots decompose the plan-structure channel only — "
                    "a feature contribution and a wording attribution come from "
                    "different models and are never summed.")

                item = labels[chosen]
                per_criterion = shap_explanations(submission["features"])
                shap_values = per_criterion.get(item.criterion.key)
                if shap_values is None:
                    st.info(
                        "The plan-structure channel has no decomposition for "
                        "this criterion, so there is no waterfall to draw. The "
                        "evidence bars on its card show what is available.",
                        icon=":material/info:")
                else:
                    render.st_waterfall(shap_values,
                                        criterion_label=item.criterion.label)
                    st.markdown("**Force plot**")
                    render.st_force_plot(shap_values)

        st.write("")
        with st.container(border=True):
            render.st_suggestions_panel(submission["suggestions"])


# ----------------------------------------------------------------------------
# Revision history
# ----------------------------------------------------------------------------
with history_tab:
    st.markdown("#### Revision history")
    st.caption(db.status())

    rows = db.submission_history(user.email)
    if not rows:
        rows = [
            {"source": entry["source"], "submitted_at": None,
             "overall_score": entry["overall_score"], "band": entry["band"]}
            for entry in st.session_state.get("session_history", [])
        ]

    if not rows:
        st.info("No submissions yet. Score a lesson plan and its revisions will "
                "be tracked here.", icon=":material/history:")
    else:
        frame = pd.DataFrame([
            {
                "Version": index + 1,
                "Plan": row["source"],
                "Submitted": (row["submitted_at"].strftime("%d %b %Y %H:%M")
                              if row.get("submitted_at") else "this session"),
                "Score": (None if row["overall_score"] is None
                          else float(row["overall_score"])),
                "Band": row["band"],
            }
            for index, row in enumerate(rows)
        ])

        left, right = st.columns([1, 1.1], gap="large")
        with left:
            st.dataframe(frame, hide_index=True, width="stretch")
            if len(frame) > 1 and frame["Score"].notna().all():
                change = frame["Score"].iloc[-1] - frame["Score"].iloc[0]
                st.metric("Change since first submission", f"{change:+.1f} points")
        with right:
            plotted = frame.dropna(subset=["Score"])
            if len(plotted) < 2:
                st.caption("A trend line appears once there are two scored "
                           "versions.")
            else:
                base = alt.Chart(plotted).encode(
                    x=alt.X("Version:O", title="Version",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Score:Q", title="Overall score (0–100)",
                            scale=alt.Scale(domain=[0, 100])),
                    tooltip=["Version", "Plan", "Score", "Band"],
                )
                st.altair_chart(
                    base.mark_line(strokeWidth=3, color=BRAND)
                    + base.mark_point(size=90, filled=True, color=BRAND),
                    width="stretch")


# ----------------------------------------------------------------------------
# Trust survey — S4's instruments
# ----------------------------------------------------------------------------
with survey_tab:
    st.markdown("#### Trust and acceptance survey")
    st.info(
        "The TAM and trust-in-AI instruments are S4's deliverable and are not "
        "wired in yet. This tab hosts them once the IRB-approved items are "
        "final; the dashboard records the pre/post response against the "
        "submission history above.",
        icon=":material/assignment:",
    )


# ----------------------------------------------------------------------------
# Transparency
# ----------------------------------------------------------------------------
with transparency_tab:
    if layer.available:
        layer.render.st_transparency_panel(get_artifacts())
    else:
        st.warning("The model layer is not loaded, so its coverage report is "
                   "unavailable.", icon=":material/warning:")

    st.divider()
    st.subheader("Deployment")
    status = artifacts_status()
    st.markdown(
        f"- **Artifacts directory:** `{status['artifacts_dir']}`\n"
        f"- **Structural model on disk:** "
        f"{'yes' if status['rubric_model'] else 'no'}\n"
        f"- **Text model on disk:** {'yes' if status['text_model'] else 'no'}\n"
        f"- **Storage:** {db.status()}\n"
        f"- **Access control:** password required · "
        f"{accounts.count_accounts()} account(s) · scrypt hashed · "
        f"{session.MAX_ATTEMPTS} attempts then a "
        f"{session.LOCKOUT_SECONDS // 60}-minute lockout · "
        f"{session.IDLE_TIMEOUT_SECONDS // 60}-minute idle timeout")
