import streamlit as st

def page_title(layout="centered"):
    st.set_page_config(
        page_title="XAI Co-Teaching Dashboard",
        page_icon="🎓",
        layout=layout,
        initial_sidebar_state="expanded",
    )