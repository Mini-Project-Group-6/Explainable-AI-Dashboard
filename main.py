# main.py — Sign-in screen
"""Entry page. Fixes carried over the first draft:

  * the password field no longer ships with a working value prefilled into it,
    and the email is a placeholder rather than a real address;
  * signing in sets session state, so ``pages/dashboard.py`` can refuse to
    render for anyone who simply types its URL;
  * the brand panel is no longer a fixed 40%-wide overlay paired with a 50%
    left padding on the form. Below ~900px that arrangement put the form
    underneath the panel, which made the page unusable on a laptop with the
    browser at half width, let alone a phone.
"""

import streamlit as st

from app import session
from app.components.title import page_title

page_title(layout="wide")

if session.current_user() is not None:
    st.switch_page("pages/dashboard.py")

BRAND = "#1D4ED8"
MUTED_LIGHT = "#5B6472"   # 6.0:1 on white; the previous #94a3b8 was 2.6:1
MUTED_DARK = "#9BA5B4"    # 7.6:1 on Streamlit's dark background

st.markdown(
    f"""
    <style>
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stSidebar"] {{ display: none; }}
    [data-testid="stSidebarCollapsedControl"] {{ display: none; }}

    .block-container {{
        max-width: 100% !important;
        padding: 10vh 8% 4rem 52% !important;
    }}

    .login-left {{
        position: fixed; top: 0; left: 0;
        width: 44%; height: 100vh;
        background: {BRAND}; color: #ffffff;
        padding: 0 clamp(24px, 4vw, 56px);
        display: flex; flex-direction: column; justify-content: center;
        box-sizing: border-box; overflow-y: auto;
    }}
    .login-left .brand {{ display:flex; align-items:center; gap:12px; margin-bottom:44px; }}
    .login-left .brand .logo {{
        background: rgba(255,255,255,0.18); width:46px; height:46px;
        border-radius:12px; display:flex; align-items:center;
        justify-content:center; font-size:22px; flex:0 0 auto;
    }}
    .login-left .brand .name {{ font-weight:700; font-size:18px; line-height:1.2; }}
    .login-left .brand .sub  {{ font-size:13px; color:rgba(255,255,255,0.82); }}
    .login-left h1 {{
        font-size: clamp(26px, 2.6vw, 40px); font-weight:800;
        line-height:1.15; margin:0 0 18px 0; color:#fff;
    }}
    .login-left p.lead {{
        font-size:16px; line-height:1.6; color:rgba(255,255,255,0.9);
        max-width:380px; margin:0 0 32px 0;
    }}
    .login-left ul {{ list-style:none; padding:0; margin:0; }}
    .login-left li {{
        font-size:15px; color:rgba(255,255,255,0.94); margin-bottom:14px;
        display:flex; align-items:center; gap:10px;
    }}
    .login-left li::before {{
        content:""; width:7px; height:7px; border-radius:50%;
        background:rgba(255,255,255,0.9); display:inline-block; flex:0 0 auto;
    }}

    /* Below this width the two-column split cannot hold: the panel becomes a
       banner above the form instead of an overlay behind it. */
    @media (max-width: 900px) {{
        .block-container {{ padding: 1.5rem 6% 3rem !important; }}
        .login-left {{
            position: static; width: 100%; height: auto;
            padding: 28px clamp(20px, 5vw, 40px); border-radius: 0 0 18px 18px;
            margin: 0 0 26px 0;
        }}
        .login-left .brand {{ margin-bottom: 22px; }}
        .login-left p.lead {{ margin-bottom: 20px; }}
    }}

    .signin-title {{ font-size:30px; font-weight:800; margin-bottom:4px; }}
    .signin-sub   {{ color:{MUTED_LIGHT}; margin-bottom:22px; font-size:15px; }}
    .signin-foot  {{ color:{MUTED_LIGHT}; font-size:12.5px; line-height:1.6;
                     margin-top:20px; }}
    @media (prefers-color-scheme: dark) {{
        .signin-sub, .signin-foot {{ color:{MUTED_DARK}; }}
    }}

    /* Border on the wrapper so the reveal-password eye sits inside the box.
       Colours are left to the theme — the draft forced a white field with near
       black text, which stayed white in dark mode. */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {{
        border:1px solid rgba(128,128,128,0.45); border-radius:10px;
        padding:2px 6px;
    }}
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
        border-color:{BRAND}; box-shadow:0 0 0 2px rgba(29,78,216,0.18);
    }}
    div[data-testid="stTextInput"] label p {{ font-weight:600; }}

    div.stButton > button {{
        background:{BRAND}; color:#fff; border:none; border-radius:10px;
        padding:0.7rem 1rem; font-weight:600; font-size:16px; margin-top:10px;
    }}
    div.stButton > button:hover {{ background:#2563EB; color:#fff; }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
            Upload lesson plans, receive rubric scores with SHAP explanations,
            and track your professional growth as a teacher.
        </p>
        <ul>
            <li>Lesson plan scoring against the NTS rubric</li>
            <li>SHAP explanation for every score</li>
            <li>Revision tracking across drafts</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="signin-title">Sign in</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signin-sub">Use your institution credentials to access the '
    'dashboard.</div>',
    unsafe_allow_html=True,
)

if st.session_state.pop("auth_expired", False):
    st.info("Your session timed out. Please sign in again.",
            icon=":material/schedule:")

has_accounts = session.accounts_exist()

if not has_accounts:
    # Deliberately says nothing about how to fix it: this page is public, and
    # the cause (no database, no accounts) is in the server log. Without it the
    # page showed only a greyed-out button.
    st.info("Sign-in isn't available at the moment. Please try again later, "
            "or contact the research team.", icon=":material/lock:")

with st.form("sign_in", border=False):
    email = st.text_input("Email address", placeholder="name@st.knust.edu.gh")
    password = st.text_input("Password", type="password",
                             placeholder="Enter your password")
    submitted = st.form_submit_button("Sign in", width="stretch",
                                      icon=":material/login:",
                                      disabled=not has_accounts)

if submitted:
    ok, message = session.sign_in(email, password)
    if ok:
        st.switch_page("pages/dashboard.py")
    else:
        st.error(message, icon=":material/error:")

# if not has_accounts:
#     # Fail closed, and say how to open it. The alternative — admitting
#     # everyone until someone remembers to configure credentials — is how the
#     # first version shipped, and it was silent about it.
#     st.warning(
#         "No accounts exist yet, so sign-in is closed. Create the first one "
#         "from a terminal on the server:",
#         icon=":material/lock:",
#     )
#     st.code("python -m app.accounts add tutor@st.knust.edu.gh --role tutor",
#             language="bash")
#     st.markdown(
#         '<div class="signin-foot">You will be prompted for a password '
#         '(minimum 10 characters). Add <code>--generate</code> to have one '
#         'generated instead.</div>',
#         unsafe_allow_html=True,
#     )
