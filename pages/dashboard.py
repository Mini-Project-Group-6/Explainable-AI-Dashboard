# pages/dashboard.py
import math

import altair as alt
import pandas as pd
import streamlit as st
from app.components.title import page_title

page_title(layout="wide")

# ----------------------------------------------------------------------------
# Theme-aware palette — adapts to Streamlit's light/dark theme
# ----------------------------------------------------------------------------
IS_DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"

# Accent / text colors that need to flip between themes
if IS_DARK:
    INK = {
        "strength": "#4ade80", "improve": "#f87171", "suggest": "#c4b5fd",
        "summary_green": "#6ee7b7", "summary_blue": "#93c5fd",
        "pill": "#93c5fd", "waterfall": "#e5e7eb",
    }
else:
    INK = {
        "strength": "#15803d", "improve": "#b91c1c", "suggest": "#6d28d9",
        "summary_green": "#065f46", "summary_blue": "#1e3a8a",
        "pill": "#1d4ed8", "waterfall": "#111827",
    }

# Translucent tints sit over either a light or dark background unchanged
TINT = {
    "strength_bg": "rgba(34,197,94,0.12)",  "strength_bd": "rgba(34,197,94,0.34)",
    "improve_bg":  "rgba(239,68,68,0.12)",  "improve_bd":  "rgba(239,68,68,0.34)",
    "suggest_bg":  "rgba(139,92,246,0.13)", "suggest_bd":  "rgba(139,92,246,0.36)",
    "summary_green_bg": "rgba(16,185,129,0.12)", "summary_green_bd": "rgba(16,185,129,0.32)",
    "summary_blue_bg":  "rgba(59,130,246,0.13)", "summary_blue_bd":  "rgba(59,130,246,0.34)",
    "pill_bg": "rgba(59,130,246,0.16)",
    "upload_bg": "rgba(148,163,184,0.08)", "upload_bd": "rgba(148,163,184,0.40)",
}

# ----------------------------------------------------------------------------
# Demo data — replace with real model output once the scoring pipeline is wired
# ----------------------------------------------------------------------------
USER = {"name": "Ama Serwah", "course": "CSM 376 · Cohort B", "initials": "AS"}

OVERALL = {"score": 80, "label": "Good — Above Average", "version": 4}

METRICS = [
    {"name": "Learning Objectives", "value": 88, "color": "#2563eb"},
    {"name": "Assessment Alignment", "value": 82, "color": "#0d9488"},
    {"name": "Pedagogical Strategy", "value": 79, "color": "#16a34a"},
    {"name": "Differentiation", "value": 65, "color": "#ea580c"},
    {"name": "Resource Integration", "value": 85, "color": "#7c3aed"},
    {"name": "Prior Knowledge", "value": 80, "color": "#0891b2"},
]

SUMMARY = (
    "This lesson plan demonstrates strong learning objectives and resource "
    "integration. Key areas for improvement include differentiation strategies "
    "and time management allocation."
)

STRENGTHS = [
    "Clear, measurable learning objectives aligned with Bloom's Taxonomy",
    "Strong formative assessment embedded throughout the lesson",
    "Effective use of real-world examples to contextualise content",
    "Well-structured introduction that activates prior knowledge",
]

IMPROVEMENTS = [
    "Differentiation strategies are limited — no provisions for learners with diverse needs",
    "Time allocations are vague and may lead to lesson overrun",
    "Summative assessment task is misaligned with stated outcomes",
    "Insufficient scaffolding for lower-ability learners",
]

SUGGESTIONS = [
    "Add tiered tasks with extension activities for advanced learners",
    "Include explicit time estimates for each lesson phase",
    "Revise the exit-ticket task to directly measure learning objectives",
    "Integrate peer-teaching opportunities to deepen understanding",
]

# SHAP feature contributions (impact on overall score)
SHAP_FEATURES = [
    {"feature": "Learning Objectives Clarity", "shap": 0.26},
    {"feature": "Assessment Alignment", "shap": 0.20},
    {"feature": "Pedagogical Strategy", "shap": 0.16},
    {"feature": "Differentiation", "shap": 0.13},
    {"feature": "Prior Knowledge Links", "shap": 0.09},
    {"feature": "Time Management", "shap": 0.05},
]

SHAP_SUMMARY = (
    "Learning Objectives Clarity and Assessment Alignment are the top positive "
    "drivers (combined SHAP +0.50). Differentiation and Time Management reduce "
    "the score by −0.09. Focus revisions on broadening differentiation "
    "strategies for the greatest score improvement."
)

# Signed per-feature SHAP contributions for this lesson plan (in score points).
# Base value (model expectation) + contributions = final output score.
SHAP_BASE = 65.0
SHAP_OUTPUT = 80.0
SHAP_CONTRIBS = [
    {"feature": "Learning Objectives Clarity", "value": 7.0},
    {"feature": "Assessment Alignment", "value": 6.0},
    {"feature": "Pedagogical Strategy", "value": 4.0},
    {"feature": "Prior Knowledge Links", "value": 3.0},
    {"feature": "Differentiation", "value": -3.0},
    {"feature": "Time Management", "value": -2.0},
]

# Diverging colors used by both SHAP plots (readable on light + dark themes).
SHAP_POS = "#ef4444"  # red  -> pushes score higher
SHAP_NEG = "#3b82f6"  # blue -> pushes score lower


def _linscale(dmin, dmax, pmin, pmax):
    span = (dmax - dmin) or 1.0
    return lambda v: pmin + (v - dmin) / span * (pmax - pmin)


def shap_waterfall_svg(base, output, contribs):
    """Proper SHAP waterfall: E[f(x)] at the bottom accumulating to f(x) at top."""
    feats = sorted(contribs, key=lambda c: abs(c["value"]), reverse=True)
    running = base
    seg = []  # accumulate bottom -> top (smallest first)
    for c in reversed(feats):
        start = running
        running += c["value"]
        seg.append({**c, "start": start, "end": running})
    disp = list(reversed(seg))  # top -> bottom for drawing

    vals = [base, output] + [s["start"] for s in seg] + [s["end"] for s in seg]
    dmin, dmax = min(vals), max(vals)
    pad = (dmax - dmin) * 0.18 or 5
    dmin, dmax = dmin - pad, dmax + pad

    W, H, left, right, top, bottom = 320, 280, 96, 304, 36, 244
    x = _linscale(dmin, dmax, left, right)
    row_h = (bottom - top) / len(disp)
    p = [f"<line x1='{x(output):.1f}' y1='{top - 8}' x2='{x(output):.1f}' y2='{bottom + 8}' "
         f"stroke='#94a3b8' stroke-width='1' stroke-dasharray='3 3'/>"]
    for i, s in enumerate(disp):
        cy = top + i * row_h + row_h / 2
        x0, x1 = x(s["start"]), x(s["end"])
        color = SHAP_POS if s["value"] > 0 else SHAP_NEG
        bh = row_h * 0.52
        p.append(f"<rect x='{min(x0, x1):.1f}' y='{cy - bh / 2:.1f}' width='{max(abs(x1 - x0), 1):.1f}' "
                 f"height='{bh:.1f}' rx='2' fill='{color}'/>")
        p.append(f"<text x='{left - 8}' y='{cy + 4:.1f}' text-anchor='end' font-size='10.5' "
                 f"fill='#94a3b8'>{s['feature']}</text>")
        sign = "+" if s["value"] > 0 else "−"
        lx = (x1 + 4) if s["value"] > 0 else (x1 - 4)
        anc = "start" if s["value"] > 0 else "end"
        p.append(f"<text x='{lx:.1f}' y='{cy + 4:.1f}' text-anchor='{anc}' font-size='10' "
                 f"fill='{color}'>{sign}{abs(s['value']):.0f}</text>")
    p.append(f"<text x='{x(base):.1f}' y='{bottom + 22}' text-anchor='middle' font-size='11' "
             f"fill='currentColor'>E[f(x)] = {base:.0f}</text>")
    p.append(f"<text x='{x(output):.1f}' y='{top - 14}' text-anchor='middle' font-size='12' "
             f"font-weight='700' fill='currentColor'>f(x) = {output:.0f}</text>")
    return (f"<div style='color:var(--text-color);'><svg width='100%' height='{H}' "
            f"viewBox='0 0 {W} {H}'>{''.join(p)}</svg></div>")


def shap_force_svg(base, output, contribs):
    """Classic SHAP force plot: red features push the score up, blue push it down."""
    pos = [c for c in contribs if c["value"] > 0]
    neg = [c for c in contribs if c["value"] < 0]
    sum_pos = sum(c["value"] for c in pos)
    sum_neg = sum(-c["value"] for c in neg)
    lo, hi = output - sum_pos, output + sum_neg
    dmin, dmax = min(lo, base), max(hi, output)
    pad = (dmax - dmin) * 0.14 or 4
    dmin, dmax = dmin - pad, dmax + pad

    W, H, left, right = 320, 170, 16, 304
    x = _linscale(dmin, dmax, left, right)
    yc, bh = 78, 34
    p = []

    def blocks(items, start, step_sign, color):
        cur = start
        for c in items:
            x0 = x(cur)
            cur += step_sign * abs(c["value"])
            x1 = x(cur)
            p.append(f"<rect x='{min(x0, x1):.1f}' y='{yc - bh / 2}' width='{max(abs(x1 - x0), 0.5):.1f}' "
                     f"height='{bh}' fill='{color}' opacity='0.92'/>")
            if abs(x1 - x0) > 30:
                p.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{yc + 4}' text-anchor='middle' "
                         f"font-size='9' fill='#fff'>{c['feature'].split()[0]}</text>")

    blocks(sorted(pos, key=lambda c: -c["value"]), lo, +1, SHAP_POS)
    blocks(sorted(neg, key=lambda c: c["value"]), output, +1, SHAP_NEG)

    p.append(f"<line x1='{x(output):.1f}' y1='{yc - bh / 2 - 10}' x2='{x(output):.1f}' "
             f"y2='{yc + bh / 2 + 6}' stroke='currentColor' stroke-width='2'/>")
    p.append(f"<text x='{x(output):.1f}' y='{yc - bh / 2 - 14}' text-anchor='middle' font-size='12' "
             f"font-weight='700' fill='currentColor'>f(x) = {output:.0f}</text>")
    p.append(f"<line x1='{x(base):.1f}' y1='{yc - bh / 2 - 2}' x2='{x(base):.1f}' "
             f"y2='{yc + bh / 2 + 16}' stroke='#94a3b8' stroke-width='1' stroke-dasharray='3 3'/>")
    p.append(f"<text x='{x(base):.1f}' y='{yc + bh / 2 + 28}' text-anchor='middle' font-size='10' "
             f"fill='#94a3b8'>base {base:.0f}</text>")
    p.append(f"<rect x='{left}' y='{H - 14}' width='10' height='10' fill='{SHAP_POS}'/>"
             f"<text x='{left + 14}' y='{H - 5}' font-size='10' fill='#94a3b8'>higher</text>")
    p.append(f"<rect x='{left + 74}' y='{H - 14}' width='10' height='10' fill='{SHAP_NEG}'/>"
             f"<text x='{left + 88}' y='{H - 5}' font-size='10' fill='#94a3b8'>lower</text>")
    return (f"<div style='color:var(--text-color); text-align:center;'><svg width='100%' height='{H}' "
            f"viewBox='0 0 {W} {H}'>{''.join(p)}</svg></div>")


# Revision history (each entry: version, date, score)
REVISIONS = [
    {"v": "v1", "date": "Oct 12, 2024", "label": "v1 — Oct 12", "score": 62},
    {"v": "v2", "date": "Oct 18, 2024", "label": "v2 — Oct 18", "score": 71},
    {"v": "v3", "date": "Oct 25, 2024", "label": "v3 — Oct 25", "score": 76},
    {"v": "v4", "date": "Nov 3, 2024", "label": "v4 — Nov 3", "score": 80},
]

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding: 3rem 2.5rem 4rem 2.5rem;
        margin: 0 auto;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.25);
    }
    /* Hide Streamlit's auto page nav ("main"/"dashboard") at the top */
    [data-testid="stSidebarNav"] { display: none; }

    /* File uploader -> dashed drop zone that fills the card */
    [data-testid="stFileUploader"] section {
        background-color: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 28px;
        width: 100%;
        min-height: 190px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stFileUploader"] button {
        background-color: #1d4ed8;
        color: #ffffff;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
    }
    [data-testid="stFileUploader"] button:hover { background-color: #2563eb; }

    .metric-row { margin-bottom: 14px; }
    .metric-head {
        display:flex; justify-content:space-between;
        font-size:15px; color:var(--text-color); font-weight:600; margin-bottom:6px;
    }
    .metric-track {
        background:rgba(128,128,128,0.18); border-radius:999px; height:8px; width:100%;
    }
    .metric-fill { height:8px; border-radius:999px; }

    .feedback-card {
        border-radius:14px; padding:18px 20px; height:100%;
    }
    .feedback-card h4 { margin:0 0 12px 0; font-size:16px; }
    .feedback-card ul { margin:0; padding-left:18px; }
    .feedback-card li { margin-bottom:10px; font-size:14px; line-height:1.45; }

    .summary-box {
        background:#ecfdf5; border:1px solid #a7f3d0; border-radius:12px;
        padding:16px 18px; font-size:14px; color:#065f46; line-height:1.5;
    }
    .summary-box.info {
        background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a;
    }
    .pill {
        background:#eff6ff; color:#1d4ed8; border-radius:999px;
        padding:4px 12px; font-size:13px; font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Theme-aware overrides (translucent tints + flipped accents)
st.markdown(
    f"""
    <style>
    .summary-box {{
        background:{TINT['summary_green_bg']}; border:1px solid {TINT['summary_green_bd']};
        color:{INK['summary_green']};
    }}
    .summary-box.info {{
        background:{TINT['summary_blue_bg']}; border:1px solid {TINT['summary_blue_bd']};
        color:{INK['summary_blue']};
    }}
    .pill {{ background:{TINT['pill_bg']}; color:{INK['pill']}; }}
    [data-testid="stFileUploader"] section {{
        background:{TINT['upload_bg']}; border:2px dashed {TINT['upload_bd']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="background:#1d4ed8; color:#fff; width:40px; height:40px;
                        border-radius:10px; display:flex; align-items:center;
                        justify-content:center; font-size:20px;">🎓</div>
            <div>
                <div style="font-weight:700; color:#3b82f6; font-size:16px;">XAI Co-Teaching</div>
                <div style="font-size:12px; color:#94a3b8;">Teacher College Portal</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px; color:#94a3b8; letter-spacing:0.08em; "
        "font-weight:700; margin-bottom:8px;'>NAVIGATION</div>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/dashboard.py", label="Dashboard", icon="📊")

    st.markdown("<div style='height:32vh;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <div style="background:#1d4ed8; color:#fff; width:38px; height:38px;
                        border-radius:50%; display:flex; align-items:center;
                        justify-content:center; font-weight:700;">{USER['initials']}</div>
            <div>
                <div style="font-weight:700; color:var(--text-color); font-size:14px;">{USER['name']}</div>
                <div style="font-size:12px; color:#94a3b8;">Student Teacher</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Logout", icon="🚪", type="tertiary"):
        st.switch_page("main.py")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
head_left, head_right = st.columns([3, 1.4], vertical_alignment="center")
with head_left:
    st.markdown(
        "<h1 style='margin-bottom:0; color:var(--text-color);'>Explainable AI Co-Teaching Dashboard</h1>"
        "<p style='color:#94a3b8; margin-top:4px;'>Teacher College Research Portal "
        "· Academic Year 2025–2026</p>",
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; justify-content:flex-end;
                    flex-wrap:nowrap; white-space:nowrap;">
            <div style="text-align:right; min-width:0;">
                <div style="font-weight:700; color:var(--text-color);">{USER['name']}</div>
                <div style="font-size:13px; color:#94a3b8;">{USER['course']}</div>
            </div>
            <div style="flex:0 0 auto; background:#1d4ed8; color:#fff; width:44px; height:44px;
                        border-radius:50%; display:flex; align-items:center;
                        justify-content:center; font-weight:700;">{USER['initials']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------
# Lesson Plan Upload (full width)
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### ⬆️ Lesson Plan Upload")
    st.file_uploader(
        "Drag & drop your lesson plan",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

st.write("")

# ----------------------------------------------------------------------------
# AI Lesson Quality Score (full width, after upload)
# ----------------------------------------------------------------------------
with st.container(border=True):
    top = st.columns([3, 1])
    with top[0]:
        st.markdown("#### ⭐ AI Lesson Quality Score")
    with top[1]:
        st.markdown(
            f"<div style='text-align:right;'><span class='pill'>Version "
            f"{OVERALL['version']}</span></div>",
            unsafe_allow_html=True,
        )

    gcol, mcol = st.columns([1, 1.4])

    with gcol:
        score = OVERALL["score"]
        # Half-circle SVG gauge (semicircle from -90deg to +90deg)
        frac = score / 100
        # arc path along a semicircle radius 70, center (80,80)
        angle = math.pi * (1 - frac)  # pi -> 0
        end_x = 80 + 70 * math.cos(angle)
        end_y = 80 - 70 * math.sin(angle)
        large = 0  # always < 180deg for semicircle portion
        st.markdown(
            f"""
            <div style="text-align:center; color:var(--text-color);">
              <svg width="180" height="110" viewBox="0 0 160 100">
                <path d="M 10 80 A 70 70 0 0 1 150 80" fill="none"
                      stroke="rgba(128,128,128,0.22)" stroke-width="14" stroke-linecap="round"/>
                <path d="M 10 80 A 70 70 0 {large} 1 {end_x:.1f} {end_y:.1f}" fill="none"
                      stroke="#2563eb" stroke-width="14" stroke-linecap="round"/>
                <text x="80" y="72" text-anchor="middle" font-size="30"
                      font-weight="700" fill="currentColor">{score}</text>
                <text x="80" y="92" text-anchor="middle" font-size="11"
                      fill="#94a3b8">out of 100</text>
              </svg>
              <div style="margin-top:6px;">
                <span class="pill">{OVERALL['label']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol:
        bars = "".join(
            f"""
            <div class="metric-row">
              <div class="metric-head"><span>{m['name']}</span><span>{m['value']}</span></div>
              <div class="metric-track">
                <div class="metric-fill" style="width:{m['value']}%; background:{m['color']};"></div>
              </div>
            </div>
            """
            for m in METRICS
        )
        st.markdown(bars, unsafe_allow_html=True)

    st.markdown(
        f"<div class='summary-box'><b>Summary:</b> {SUMMARY}</div>",
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------
# AI Feedback
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### 💡 AI Feedback")

    def feedback_block(title, items, bg, border, color):
        lis = "".join(f"<li>{i}</li>" for i in items)
        return (
            f"<div class='feedback-card' style='background:{bg}; border:1px solid {border};'>"
            f"<h4 style='color:{color};'>{title}</h4><ul>{lis}</ul></div>"
        )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            feedback_block("📈 Strengths", STRENGTHS,
                           TINT["strength_bg"], TINT["strength_bd"], INK["strength"]),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            feedback_block("📉 Areas for Improvement", IMPROVEMENTS,
                           TINT["improve_bg"], TINT["improve_bd"], INK["improve"]),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            feedback_block("💡 Improvement Suggestions", SUGGESTIONS,
                           TINT["suggest_bg"], TINT["suggest_bd"], INK["suggest"]),
            unsafe_allow_html=True,
        )

st.write("")

# ----------------------------------------------------------------------------
# Explainable AI — SHAP Analysis
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### ℹ️ Explainable AI — SHAP Analysis")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-6px;'>SHAP (SHapley Additive exPlanations) "
        "values show how each rubric feature contributed to the overall quality score.</p>",
        unsafe_allow_html=True,
    )

    # --- Feature Importance (full-width horizontal bars) ---
    st.markdown("**Feature Importance**")
    fi = pd.DataFrame(SHAP_FEATURES)
    fi_chart = (
        alt.Chart(fi)
        .mark_bar(color="#2563eb", cornerRadiusEnd=3, size=22)
        .encode(
            x=alt.X("shap:Q", title=None, axis=alt.Axis(values=[0, 0.14, 0.28])),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", "shap"],
        )
        .properties(height=240)
    )
    st.altair_chart(fi_chart, use_container_width=True)

    # --- SHAP Waterfall + Force Plot on one row ---
    wf_col, force_col = st.columns(2, gap="large")
    with wf_col:
        st.markdown("**SHAP Waterfall**")
        st.markdown(
            shap_waterfall_svg(SHAP_BASE, SHAP_OUTPUT, SHAP_CONTRIBS),
            unsafe_allow_html=True,
        )
    with force_col:
        st.markdown("**SHAP Force Plot**")
        st.markdown(
            shap_force_svg(SHAP_BASE, SHAP_OUTPUT, SHAP_CONTRIBS),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='summary-box info'><b>Explanation Summary:</b> {SHAP_SUMMARY}</div>",
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------
# Revision History
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### 📈 Revision History")

    r_left, r_right = st.columns([1.1, 1], gap="large")

    with r_left:
        prev = None
        rows = ""
        for i, rev in enumerate(REVISIONS):
            delta = ""
            if prev is not None:
                diff = rev["score"] - prev
                delta = (f"<span style='color:#16a34a; font-size:13px; font-weight:600;'>"
                         f"⌃ +{diff} pts</span> "
                         f"<span style='color:#94a3b8; font-size:13px;'>{rev['date']}</span>")
            else:
                delta = f"<span style='color:#94a3b8; font-size:13px;'>{rev['date']}</span>"
            prev = rev["score"]
            active = i == len(REVISIONS) - 1
            dot_bg = "#1d4ed8" if active else "#94a3b8"
            line = ("<div style='width:2px; height:26px; background:rgba(128,128,128,0.3); "
                    "margin-left:18px;'></div>") if not active else ""
            rows += f"""
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <div>
                    <div style="background:{dot_bg}; color:#fff; width:38px; height:38px;
                                border-radius:50%; display:flex; align-items:center;
                                justify-content:center; font-weight:700; font-size:13px;">{rev['v']}</div>
                </div>
                <div style="flex:1;">
                    <div style="font-weight:700; color:var(--text-color);">{rev['label']}</div>
                    <div style="margin-top:2px;">{delta}</div>
                </div>
                <div><span class="pill">{rev['score']}/100</span></div>
            </div>
            {line}
            """
        st.markdown(rows, unsafe_allow_html=True)

    with r_right:
        st.markdown("**Improvement Trend**")
        tr = pd.DataFrame(REVISIONS)
        order = list(tr["label"])
        base = alt.Chart(tr).encode(
            x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("score:Q", title=None,
                    scale=alt.Scale(domain=[50, 100]),
                    axis=alt.Axis(values=[50, 65, 80, 100])),
        )
        trend = (
            base.mark_line(color="#2563eb", strokeWidth=3)
            + base.mark_point(color="#2563eb", size=90, filled=True)
        ).properties(height=260)
        st.altair_chart(trend, use_container_width=True)
