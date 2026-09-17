import os
from datetime import date
from typing import Optional

import streamlit as st

#: Month the academic year starts in. An assumption, so it can be overridden
#: outright with COTEACH_ACADEMIC_YEAR.
ACADEMIC_YEAR_START_MONTH = 9


def page_title(layout="centered"):
    st.set_page_config(
        page_title="XAI Co-Teaching Dashboard",
        page_icon="🎓",
        layout=layout,
        initial_sidebar_state="expanded",
    )


def academic_year(today: Optional[date] = None) -> str:
    """"2026–2027" for any date from September 2026 to August 2027.

    The header used to hard-code "2025–2026", which was wrong from the first
    day of the next year.
    """
    configured = os.getenv("COTEACH_ACADEMIC_YEAR", "").strip()
    if configured:
        return configured
    today = today or date.today()
    first = today.year if today.month >= ACADEMIC_YEAR_START_MONTH else today.year - 1
    return f"{first}–{first + 1}"
