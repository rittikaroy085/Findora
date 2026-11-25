import base64
import io
import os
import random
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from findora_engine import Findora, build_index, load_index

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Findora | AI Gallery Search",
    page_icon="assets/icon-64.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# -----------------------------
# CONSTANTS AND HELPERS
# -----------------------------

# Material Symbols: one crisp, consistent vector icon set built into Streamlit
PAGES = [
    ":material/home: Home",
    ":material/photo_library: My Gallery",
    ":material/settings: Settings",
]
GRID_COLUMNS = 4
PAGE_SIZE = 12
SUGGESTIONS = [
    "Handwritten notes",
    "Beach sunset",
    "Birthday cake",
    "Magenta flower",
    "Mountain trek",
    "Pet portrait",
]


@st.cache_resource(show_spinner="Loading AI models (first time only)...")
def get_engine():
    return Findora()


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str:
    try:
        data = Path("assets/logo.png").read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    except Exception:
        return ""


@st.cache_data(show_spinner=False, max_entries=512)
def load_thumbnail(path: str, mtime: float, size: int = 480):
    with Image.open(path) as img:
        thumb = ImageOps.exif_transpose(img).convert("RGB")
    thumb.thumbnail((size, size))
    return thumb


@st.cache_data(show_spinner=False)
def masonry_html(paths: tuple) -> str:
    """Fading masonry preview built from small inline thumbnails."""
    tiles = []
    for p in paths:
        try:
            with Image.open(p) as img:
                im = ImageOps.exif_transpose(img).convert("RGB")
            im.thumbnail((360, 360))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
            b64 = base64.b64encode(buf.getvalue()).decode()
            tiles.append(f'<img src="data:image/jpeg;base64,{b64}" alt="">')
        except Exception:
            continue
    if not tiles:
        return ""
    # one line, no indentation: Markdown would treat indented HTML as code
    return '<div class="masonry">' + "".join(tiles) + "</div>"


def show_image(image, caption=None):
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
                    st.caption(f"Can't open {Path(path).name}")
                    continue
                show_image(thumb, caption)


def go_to(page: str):
    st.session_state.page = page


def use_chip(text: str):
    st.session_state.query_text = text
    st.session_state.run_search = True


def show_more():
    results = st.session_state.results
    if results:
        st.session_state.top_k += PAGE_SIZE
        st.session_state.query_text = results["query"]
        st.session_state.run_search = True


def clear_results():
    st.session_state.results = None


def do_search(query: str):
    query = query.strip()
    library = st.session_state.library
    if not query:
        st.warning("Describe what you remember first, for example: a magenta flower.")
        return
    if library is None:
        st.info("Build your searchable library first by choosing a gallery folder.")
        st.button(
            "Choose folder",
            key="search_needs_index",
            icon=":material/folder_open:",
            on_click=go_to,
            args=(PAGES[1],),
        )
        return
    prev = st.session_state.results
    if prev is None or prev["query"] != query:
        st.session_state.top_k = PAGE_SIZE
    engine = get_engine()
    hits = engine.search(query, library.vectors, st.session_state.top_k)
    st.session_state.results = {
        "query": query,
        "hits": [(library.paths[i], score) for i, score in hits],
    }


# -----------------------------
# SESSION STATE
# -----------------------------

if "library" not in st.session_state:
    st.session_state.library = load_index()  # reuse a saved index if present
st.session_state.setdefault("results", None)
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("top_k", PAGE_SIZE)
st.session_state.setdefault("query_text", "")

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    logo = logo_data_uri()
    logo_tag = f'<img src="{logo}" alt="">' if logo else ""
    st.markdown(
        '<div class="brandrow">'
        f"{logo_tag}"
        '<div><div class="brand">Findora</div>'
        '<div class="tagline">Search by memory, not metadata.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        PAGES,
        key="page",
        label_visibility="collapsed",
    )

    library = st.session_state.library
    if library is None:
        st.markdown(
            '<div class="pill off"><span class="dot"></span>No library yet</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="pill"><span class="dot"></span>{len(library)} images indexed</div>',
            unsafe_allow_html=True,
        )
        st.caption(Path(library.folder).name)

    st.caption("Findora • Version 1.0")

# -----------------------------
# HOME PAGE
# -----------------------------

if page == PAGES[0]:

    st.markdown(
        '<div class="hero">'
        '<h1 class="hero-title">Search your <span class="grad">memories</span></h1>'
        '<p class="hero-sub">Describe a photo the way you remember it. '
        "No filenames, no endless scrolling.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Pill-shaped hero search bar with the action button inside on the right
    with st.form("search_form", border=False):
        st.text_input(
            "Search your memories",
            key="query_text",
            placeholder="A magenta flower on a white background...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search", icon=":material/arrow_forward:")

    # Clickable suggestion chips
    try:
        chip_row = st.container(
            horizontal=True, horizontal_alignment="center", key="chips"
        )
    except TypeError:  # older Streamlit without horizontal containers
        chip_row = st.container(key="chips")
    with chip_row:
        for i, text in enumerate(SUGGESTIONS):
            st.button(text, key=f"chip_{i}", on_click=use_chip, args=(text,))

    run_flag = st.session_state.pop("run_search", False)
    if submitted or run_flag:
        do_search(st.session_state.query_text)

    st.markdown("")

    results = st.session_state.results
    library = st.session_state.library

    if results:
        head, clear = st.columns([5, 1], vertical_alignment="center")
        with head:
            st.markdown(f"### Results for “{results['query']}”")
            st.caption(
                f"Top {len(results['hits'])} matches from {len(library)} images. "
                "Higher scores are better; good matches typically score 0.25 to 0.35."
            )
        with clear:
            st.button(
                "Clear",
                key="clear_results",
                icon=":material/close:",
                on_click=clear_results,
            )
        render_grid([
            (path, f"{rank} · {score:.3f} · {Path(path).name}")
            for rank, (path, score) in enumerate(results["hits"], 1)
        ])
        if st.session_state.top_k < len(library):
            st.button(
                "Show more",
                key="show_more",
                icon=":material/expand_more:",
                on_click=show_more,
            )
    else:
        html = ""
        if library is not None:
            sample = random.Random(7).sample(library.paths, min(16, len(library)))
            html = masonry_html(tuple(sample))

        if html:
            st.markdown(html, unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown("### :material/add_photo_alternate: Bring your photos in")
                st.write(
                    "Choose a folder and Findora indexes it on your device. "
                    "Then search by describing what you remember."
                )
                st.button(
                    "Choose folder",
                    key="choose_folder",
                    icon=":material/folder_open:",
                    on_click=go_to,
                    args=(PAGES[1],),
                )

# -----------------------------
# GALLERY PAGE
# -----------------------------

elif page == PAGES[1]:

    st.markdown("## My Gallery")

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
        placeholder=r"C:\Users\you\Pictures",
    )

    if st.button(
        "Build / update index",
        icon=":material/bolt:",
        use_container_width=True,
    ):
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

    st.markdown("## Settings")
    st.write("Manage your Findora experience.")

    st.markdown("### :material/lock: Privacy")
    st.write("Findora is designed to process your selected images locally.")
