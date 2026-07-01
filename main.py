# main.py — Sign-in screen
import streamlit as st
from app.components.title import page_title

page_title(layout="wide")

# ----------------------------------------------------------------------------
# Styling: split-screen login (fixed blue panel left, form right)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Strip default chrome so the panel can go full-bleed */
    [data-testid="stHeader"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stAppViewContainer"] { background: var(--secondary-background-color); }

    .block-container {
        max-width: 100% !important;
        padding: 14vh 8% 4rem 50% !important;
    }

    /* Left brand panel */
    .login-left {
        position: fixed; top: 0; left: 0;
        width: 40%; height: 100vh;
        background: #1d4ed8; color: #ffffff;
        padding: 0 56px;
        display: flex; flex-direction: column; justify-content: center;
        box-sizing: border-box;
    }
    .login-left .brand { display:flex; align-items:center; gap:12px; margin-bottom:48px; }
    .login-left .brand .logo {
        background: rgba(255,255,255,0.18); width:46px; height:46px;
        border-radius:12px; display:flex; align-items:center; justify-content:center;
        font-size:22px;
    }
    .login-left .brand .name { font-weight:700; font-size:18px; line-height:1.2; }
    .login-left .brand .sub  { font-size:13px; color:rgba(255,255,255,0.75); }
    .login-left h1 {
        font-size:40px; font-weight:800; line-height:1.15; margin:0 0 20px 0; color:#fff;
    }
    .login-left p.lead {
        font-size:16px; line-height:1.6; color:rgba(255,255,255,0.85);
        max-width:380px; margin:0 0 36px 0;
    }
    .login-left ul { list-style:none; padding:0; margin:0; }
    .login-left li {
        font-size:15px; color:rgba(255,255,255,0.9); margin-bottom:16px;
        display:flex; align-items:center; gap:10px;
    }
    .login-left li::before {
        content:""; width:7px; height:7px; border-radius:50%;
        background:rgba(255,255,255,0.85); display:inline-block;
    }

    /* Form (right side) */
    .signin-title { font-size:30px; font-weight:800; color:var(--text-color); margin-bottom:4px; }
    .signin-sub   { color:#94a3b8; margin-bottom:28px; font-size:15px; }

    /* Border/background on the wrapper so the password eye sits inside the box */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        background:#ffffff;
        border:1px solid #cbd5e1; border-radius:10px;
        padding:4px 8px 4px 6px;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color:#1d4ed8; box-shadow:0 0 0 2px rgba(29,78,216,0.15);
    }
    div[data-testid="stTextInput"] input {
        background:transparent; color:#0f172a;
        border:none; padding:8px 8px; height:auto;
    }
    /* Reveal-password eye button */
    div[data-testid="stTextInput"] button[data-baseweb="button"],
    div[data-testid="stTextInput"] div[data-baseweb="input"] button {
        background:transparent; color:#64748b; border:none;
    }
    div[data-testid="stTextInput"] label p { font-weight:600; color:var(--text-color); }

    div.stButton > button {
        background:#1d4ed8; color:#fff; border:none; border-radius:10px;
        padding:0.7rem 1rem; font-weight:600; font-size:16px; margin-top:8px;
    }
    div.stButton > button:hover { background:#2563eb; color:#fff; }

    .signin-footer {
        text-align:center; color:#94a3b8; font-size:13px;
        line-height:1.6; margin-top:22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Left brand panel
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="login-left">
        <div class="brand">
            <div class="logo">🎓</div>
            <div>
                <div class="name">XAI Co-Teaching</div>
                <div class="sub">Teacher College Portal</div>
            </div>
        </div>
        <h1>Explainable AI for<br>Teacher Education</h1>
        <p class="lead">
            Upload lesson plans, receive AI-powered quality scores with SHAP
            explanations, and track your professional growth as a teacher.
        </p>
        <ul>
            <li>AI Lesson Quality Scoring</li>
            <li>SHAP Explainability</li>
            <li>Revision Tracking</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sign-in form (right side)
# ----------------------------------------------------------------------------
st.markdown('<div class="signin-title">Sign in</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signin-sub">Use your institution credentials to access the dashboard.</div>',
    unsafe_allow_html=True,
)

st.text_input("Email address", value="amaserwah@knust.edu.gh", placeholder="Enter your email address")
st.text_input("Password", value="1234", type="password", placeholder="Enter your password")

if st.button("Sign In", width="stretch", icon="➡️"):
    st.switch_page("pages/dashboard.py")

st.markdown(
    """
    <div class="signin-footer">
        Kwame Nkrumah University of Science and Technology · College of Science<br>
        Research Ethics Clearance: KNUST-SCI-2026-047
    </div>
    """,
    unsafe_allow_html=True,
)
