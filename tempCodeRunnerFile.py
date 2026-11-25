from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from findora_engine import Findora, build_index, load_index

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
# CONSTANTS AND HELPERS
# -----------------------------

PAGES = ["🏠 Home", "🖼️ My Gallery", "⚙️ Settings"]
GRID_COLUMNS = 4


@st.cache_resource(show_spinner="Loading AI models (first time only)...")
def get_engine():
    return Findora()


@st.cache_data(show_spinner=False, max_entries=512)
def load_thumbnail(path: str, mtime: float, size: int = 480):
    with Image.open(path) as img:
        thumb = ImageOps.exif_transpose(img).convert("RGB")
    thumb.thumbnail((size, size))
    return thumb


def show_image(image, caption=None):
    # Newer Streamlit uses width="stretch"; older versions use use_container_width.
    try:
        st.image(image, caption=caption, width="stretch")
    except Exception:
        st.image(image, caption=caption, use_container_width=True)


def render_grid(items):
    """items: list of (image_path, caption)."""
    for start in range(0, len(items), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (path, caption) in zip(cols, items[start:start + GRID_COLUMNS]):
            with col:
                try:
                    thumb = load_thumbnail(path, os.path.getmtime(path))
                except Exception:
                    st.caption(f"⚠️ Can't open {Path(path).name}")
                    continue
                show_image(thumb, caption)


def go_to(page: str):
    st.session_state.page = page


# -----------------------------
# SESSION STATE
# -----------------------------

if "library" not in st.session_state:
    st.session_state.library = load_index()  # reuse a saved index if present
st.session_state.setdefault("results", None)
st.session_state.setdefault("page", PAGES[0])

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
        PAGES,
        key="page",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### LIBRARY")
    library = st.session_state.library
    if library is None:
        st.caption("No folder selected yet.")
    else:
        st.caption(f"📁 {Path(library.folder).name}")
        st.caption(f"{len(library)} images indexed")

    st.markdown("---")

    st.caption("AI-powered visual discovery")
    st.caption("Findora • Version 1.0")

# -----------------------------
# HOME PAGE
# -----------------------------

if page == PAGES[0]:

    st.markdown("## Welcome to Findora ✨")
