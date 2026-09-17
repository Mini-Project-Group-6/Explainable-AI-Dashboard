# app/components/sidebar.py
"""The dashboard sidebar: brand, navigation, signed-in user, sign out."""

from __future__ import annotations

import html

import streamlit as st

from app import session

#: Muted text. #94a3b8 — used throughout the first draft — is 2.6:1 on a white
#: background, well under the 4.5:1 WCAG AA floor, so secondary text was close
#: to unreadable in light mode. These two clear AA in their own theme.
MUTED_LIGHT = "#5B6472"
MUTED_DARK = "#9BA5B4"

BRAND = "#1D4ED8"


def muted() -> str:
    dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    return MUTED_DARK if dark else MUTED_LIGHT


def dashboard_sidebar(user: session.User) -> None:
    """Render the sidebar. Returns nothing; the caller owns page content."""
    grey = muted()
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <div style="background:{BRAND};color:#fff;width:40px;height:40px;
                            border-radius:10px;display:flex;align-items:center;
                            justify-content:center;font-size:20px;">🎓</div>
                <div>
                    <div style="font-weight:700;font-size:16px;
                                color:var(--text-color);">XAI Co-Teaching</div>
                    <div style="font-size:12px;color:{grey};">Teacher College Portal</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            f"<div style='font-size:12px;color:{grey};letter-spacing:0.08em;"
            f"font-weight:700;margin-bottom:8px;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/dashboard.py", label="Dashboard", icon=":material/dashboard:")

        st.divider()
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="background:{BRAND};color:#fff;width:38px;height:38px;
                            border-radius:50%;display:flex;align-items:center;
                            justify-content:center;font-weight:700;flex:0 0 auto;"
                     >{html.escape(user.initials)}</div>
                <div style="min-width:0;">
                    <div style="font-weight:700;color:var(--text-color);font-size:14px;
                                overflow:hidden;text-overflow:ellipsis;"
                         >{html.escape(user.name)}</div>
                    <div style="font-size:12px;color:{grey};">{html.escape(user.role_label)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign out", icon=":material/logout:", type="tertiary",
                     width="stretch"):
            session.sign_out()
            st.switch_page("main.py")
