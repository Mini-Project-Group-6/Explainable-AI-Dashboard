import streamlit as st

st.set_page_config(
    page_title="XAI Co-Teaching Dashboard",
    page_icon="🎓",
)


st.sidebar.markdown(
    """
    <h3 style="font-size: 22px; font-family: 'Roboto Slab', serif; font-weight: 700; text-align: left; margin-top: 70px; color: white;">🎓 XAI Co-Teaching Dashboard</h3>
    """,
    unsafe_allow_html=True)


st.sidebar.markdown(
    """
    <p style="font-size:15px; color:rgba(255,255,255,0.75);margin:20px; max-width:300px;">
    Upload lesson plans, receive AI-powered quality scores with SHAP explanations,
    and track your professional growth as a teacher.
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <ul style="font-size:15px; color:rgba(255,255,255,0.75); line-height:1.8; margin:20px; ">
        <li>Upload lesson plans</li>
        <li>AI Lesson Quality Scoring</li>
        <li>SHAP Explainability</li>
        <li>Revision Tracking</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #1d4ed8;
        min-width: 400px;
        max-width: 400px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.header("Sign In")
st.caption("Use your institution credentials to access the dashboard.")

st.text_input("Email address", placeholder="Enter your email address", value="ama@knust.edu.gh")
st.text_input("Password", placeholder="Enter your password", value="1234", type="password")


st.markdown("""
<style>
div.stButton > button {
    background-color: #1d4ed8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
}
""", unsafe_allow_html=True)

st.button("Sign In", width='stretch', icon="🚀")