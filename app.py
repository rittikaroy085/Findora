
import streamlit as st
from pathlib import Path

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Findora | AI Gallery Search",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD CUSTOM CSS
# -----------------------------

css_path = Path("assets/style.css")

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.markdown("# 🌌 Findora")
    st.caption("Search by memory, not metadata.")

    st.markdown("---")

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🖼️ My Gallery", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### LIBRARY")
    st.caption("No folder selected yet.")

    st.markdown("---")

    st.caption("AI-powered visual discovery")
    st.caption("Findora • Version 1.0")

# -----------------------------
# HOME PAGE
# -----------------------------

if page == "🏠 Home":

    st.markdown("## Welcome to Findora ✨")

    st.markdown(
        "### Your memories are only a search away."
    )

    st.write(
        "Find any image using the words that describe it. "
        "No filenames. No endless scrolling."
    )

    st.markdown("")

    query = st.text_input(
        "Search your memories",
        placeholder="Try: A magenta flower with a white background..."
    )

    st.button("🔍 Search memories", use_container_width=True)

    st.markdown("---")

    st.markdown("### Get started")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📁 Select your gallery")
            st.write(
                "Choose the folder containing your images "
                "to build your searchable library."
            )
            st.button("Choose folder", key="choose_folder")

    with col2:
        with st.container(border=True):
            st.markdown("### 🧠 AI-powered search")
            st.write(
                "Describe what you remember. "
                "Findora will search for visually matching images."
            )
            st.button("Explore search", key="explore_search")

# -----------------------------
# GALLERY PAGE
# -----------------------------

elif page == "🖼️ My Gallery":

    st.markdown("## My Gallery 🖼️")
    st.write("Your indexed images will appear here.")

    st.info("Select a gallery folder to get started.")

# -----------------------------
# SETTINGS PAGE
# -----------------------------

elif page == "⚙️ Settings":

    st.markdown("## Settings ⚙️")
    st.write("Manage your Findora experience.")

    st.markdown("### Privacy")
    st.write(
        "Findora is designed to process your selected images locally."
    )
