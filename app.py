import os
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

    st.markdown(
        "### Your memories are only a search away."
    )

    st.write(
        "Find any image using the words that describe it. "
        "No filenames. No endless scrolling."
    )

    st.markdown("")

    with st.form("search_form", border=False):
        query = st.text_input(
            "Search your memories",
            placeholder="Try: A magenta flower with a white background..."
        )
        with st.expander("Search options"):
            top_k = st.slider("Number of results", 4, 40, 12, step=4)
        submitted = st.form_submit_button(
            "🔍 Search memories", use_container_width=True
        )

    if submitted:
        library = st.session_state.library
        if not query.strip():
            st.warning("Describe what you remember first, for example: a magenta flower.")
        elif library is None:
            st.info("Build your searchable library first by choosing a gallery folder.")
            st.button(
                "Choose folder",
                key="search_needs_index",
                on_click=go_to,
                args=(PAGES[1],),
            )
        else:
            engine = get_engine()
            hits = engine.search(query.strip(), library.vectors, top_k)
            st.session_state.results = {
                "query": query.strip(),
                "hits": [(library.paths[i], score) for i, score in hits],
            }

    st.markdown("---")

    results = st.session_state.results

    if results:
        st.markdown(f"### Results for “{results['query']}”")
        st.caption(
            f"Top {len(results['hits'])} matches from "
            f"{len(st.session_state.library)} images. Higher scores are better; "
            "good matches typically score around 0.25 to 0.35."
        )
        render_grid([
            (path, f"{rank} · score {score:.3f} · {Path(path).name}")
            for rank, (path, score) in enumerate(results["hits"], 1)
        ])
    else:
        st.markdown("### Get started")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("### 📁 Select your gallery")
                st.write(
                    "Choose the folder containing your images "
                    "to build your searchable library."
                )
                st.button(
                    "Choose folder",
                    key="choose_folder",
                    on_click=go_to,
                    args=(PAGES[1],),
                )

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

elif page == PAGES[1]:

    st.markdown("## My Gallery 🖼️")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    library = st.session_state.library
    default_folder = st.session_state.get("last_folder") or (
        library.folder if library else ""
    )

    folder_text = st.text_input(
        "Gallery folder",
        value=default_folder,
        placeholder=r"C:\Users\you\Pictures"
    )

    if st.button("⚡ Build / update index", use_container_width=True):
        folder = Path(folder_text.strip().strip('"'))
        if not folder.is_dir():
            st.error("That folder doesn't exist. Check the path and try again.")
        else:
            st.session_state.last_folder = str(folder)
            engine = get_engine()
            bar = st.progress(0.0, text="Starting...")

            def update(done, total, name):
                bar.progress(done / total, text=f"Indexing {done}/{total}: {name}")

            try:
                index, stats = build_index(engine, folder, update)
            except ValueError as err:
                st.error(str(err))
            else:
                st.session_state.library = index
                st.session_state.results = None
                st.session_state.flash = (
                    f"Indexed {len(index)} images "
                    f"({stats['new']} new, {stats['reused']} unchanged, "
                    f"{stats['skipped']} skipped)."
                )
                st.rerun()

    library = st.session_state.library

    if library is None:
        st.info("Select a gallery folder to get started.")
    else:
        st.caption(f"{len(library)} images indexed from {library.folder}")
        shown = st.slider("Images to show", 8, 60, 24, step=4)
        render_grid([(p, Path(p).name) for p in library.paths[:shown]])

# -----------------------------
# SETTINGS PAGE
# -----------------------------

elif page == PAGES[2]:

    st.markdown("## Settings ⚙️")
    st.write("Manage your Findora experience.")

    st.markdown("### Privacy")
    st.write(
        "Findora is designed to process your selected images locally."
    )
