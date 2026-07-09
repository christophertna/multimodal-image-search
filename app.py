"""
app.py is the Streamlit web interface for the local multimodal image search engine

How to run:
    streamlit run app.py

Architecture note:

    app.py is the outermost layer of the project. It also owns 1 responsibility: user interaction
    (imports from embedder.py & indexer.py but doesnt contain the model or database logic)

    Streamlit re-runs this entire script from top to bottom on every user
    interaction (button click, text input, slider change)
    
    This is different from a normal Python script — think of it as a reactive loop
    `st.session_state` is the mechanism for preserving values across re-runs
"""

# Streamlit is a Python library specifically designed to turn data scripts into interactive, web-based applications
# Basically a bridge between backend logic (Python code) & visual interface that users can interact with
# Without it, pretty much forced to use the CLI for all

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

# recall from __init__.py imports:
from src.embedder import CLIPEmbedder
from src.indexer import VectorIndexer

# Constants
DEFAULT_INDEX_DIR = "./index"
DEFAULT_DATA_DIR  = "./data/images"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Light/dark palettes for the sidebar theme toggle.
# `config.toml` only sets the theme ONCE at startup, so to let the user flip
# modes live from a widget we override colors ourselves with a small <style>
# block (see apply_theme() below) rather than relying on config.toml alone.
LIGHT_THEME = {
    "background": "#F2F0EA",
    "secondary_background": "#E3E0D8",
    "text": "#000000",
    "border": "#000000",
    "primary": "#5B6C8F",
    "input_background": "#FFFFFF",
    "input_text": "#000000",
}
DARK_THEME = {
    "background": "#1E1E1E",
    "secondary_background": "#2A2A2A",
    "text": "#FFFFFF",
    "border": "#FFFFFF",
    "primary": "#5C7CFA",
    "input_background": "#333333",
    "input_text": "#FFFFFF",
}

# Cached resource loading:
# @st.cache_resource tells Streamlit to run this function ONCE & reuse the returned object across all re-runs and all users
# (basically the persistent memory of the application)

# Normally, when you interact with a Streamlit app, Streamlit re-runs your entire script from top to bottom.
# Without caching, every single button click would force the computer to:
# 1- Reload the CLIP model from the hard drive into RAM
# 2- Re-initialize the connection to the ChromaDB database
# 3- Wait 5–10 seconds

# This would be terrible and annoying, so the solution is to have it in the cache (reuse the objects)

# This is the correct decorator for heavyweight objects like ML models and database connections
# (Different from @st.cache_data, which is for cacheing serializable data like dataframes or API responses)
@st.cache_resource
def load_embedder() -> CLIPEmbedder:
    """Load the CLIP model once and cache it for the session."""
    return CLIPEmbedder()


@st.cache_resource
def load_indexer(index_dir: str) -> VectorIndexer:
    """Connect to ChromaDB once and cache the connection for the session."""
    return VectorIndexer(persist_dir=index_dir)


def apply_theme(dark: bool) -> None:
    """
    Inject CSS to handle all theme overrides, including the sidebar collapse button.
    """
    theme = DARK_THEME if dark else LIGHT_THEME

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}
        [data-testid="stHeader"] {{
            background-color: {theme["background"]} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {theme["secondary_background"]} !important;
        }}
        /* Sidebar collapse/expand button fix. stSidebarCollapsedControl
           (and targeting it as an <svg>) no longer exist in current
           Streamlit — confirmed via live DOM inspection. The collapse
           ("<<", shown inside the open sidebar) and expand (">>", shown in
           the main content when the sidebar is closed) are two separate
           real testids, and the icon is a Material Symbols font ligature,
           not an <svg>, hence no svg-fill rule needed anymore. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stExpandSidebarButton"],
        [data-testid="stExpandSidebarButton"] span {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stSidebar"][data-testid="stSidebar"][data-testid="stSidebar"] * {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {{
            background-color: {theme["input_background"]} !important;
            color: {theme["input_text"]} !important;
        }}
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {{
            color: {theme["input_text"]} !important;
            opacity: 0.6;
        }}
        /* The raw <input>/<textarea> above gets our background, but Streamlit
           wraps it in an outer root element that keeps its own separate,
           un-themed background/border. Since the inner input doesn't fill
           that box exactly, the old color shows through as a thin sliver
           above/below it — reading as a stray "extra border". Theming the
           root element too removes the mismatch. (stTextInputRootElement is
           the current real testid — older data-baseweb selectors no longer
           exist in newer Streamlit versions and matched nothing.) */
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextAreaRootElement"] {{
            background-color: {theme["input_background"]} !important;
            border-color: {theme["border"]} !important;
        }}
        
        /* stVerticalBlockBorderWrapper no longer exists in current Streamlit
           (confirmed via live DOM inspection) — the border/background for
           st.container(border=True) now live directly on the plain
           stVerticalBlock testid instead, shared with every other vertical
           block on the page. We only touch border-color (not width/style)
           so untouched, non-bordered blocks elsewhere aren't affected —
           a 0-width border is invisible no matter its color. The old
           selector is kept alongside for older Streamlit versions that
           still use it. */
        /* The full-width line under the tab bar is a ::after pseudo-element
           on the tablist (found via live DOM inspection: background-color
           rgba(49, 51, 63, 0.1), a dark gray at 10% opacity) — invisible
           against a dark background since it never lightens with the mode. */
        [data-testid="stTabs"] [role="tablist"]::after {{
            background-color: {theme["border"]} !important;
            opacity: 0.3;
        }}
        [data-testid="stVerticalBlock"] {{
            border-color: {theme["border"]} !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"][data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {theme["secondary_background"]} !important;
            border: 1px solid {theme["border"]} !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"][data-testid="stVerticalBlockBorderWrapper"] *,
        [data-testid="stMetricLabel"][data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"][data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"][data-testid="stMetricValue"],
        [data-testid="stMetricValue"][data-testid="stMetricValue"] *,
        [data-testid="stCaptionContainer"][data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"][data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"][data-testid="stSidebar"] small,
        [data-testid="stSidebar"][data-testid="stSidebar"] small *,
        [data-testid="stSidebar"][data-testid="stSidebar"] p,
        [data-testid="stSidebar"][data-testid="stSidebar"] span {{
            color: {theme["text"]} !important;
        }}
        body .stTabs [data-baseweb="tab"],
        body .stTabs [data-baseweb="tab"] * {{
            color: {theme["text"]} !important;
        }}
        /* Tab labels ("Search" / "Index Images") — data-baseweb doesn't exist
           in current Streamlit versions (confirmed via live DOM inspection),
           so the old selector matched nothing and tabs kept Streamlit's
           native colors (a fixed red accent on the active tab, fixed gray
           on inactive ones) regardless of mode. stTab is the real testid. */
        /* Inline markdown code spans (e.g. `./data/images`) use Streamlit's
           own fixed white background + green syntax color by default,
           unrelated to the app's theme — confirmed via live DOM inspection
           (rgb(248, 249, 251) bg / rgb(21, 130, 55) text, always, both modes). */
        code {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
        }}
        [data-testid="stTab"],
        [data-testid="stTab"] p {{
            color: {theme["text"]} !important;
        }}
        /* The active-tab underline is a React Aria internal element with no
           data-testid — found via live DOM inspection (class
           "react-aria-SelectionIndicator"). It's Streamlit's fixed red
           accent by default, so it's re-themed to the app's primary color
           to move with dark/light mode too. */
        .react-aria-SelectionIndicator {{
            background-color: {theme["primary"]} !important;
        }}
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"],
        .stButton > button[data-testid="baseButton-primary"] {{
            background-color: {theme["primary"]} !important;
            border-color: {theme["primary"]} !important;
            color: white !important;
            transition: filter 0.15s ease, transform 0.15s ease !important;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {{
            filter: brightness(1.12);
            transform: translateY(-1px);
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[data-testid="stBaseButton-primary"] p,
        .stButton > button[data-testid="baseButton-primary"] p {{
            color: white !important;
        }}
        [data-testid="stAlert"] {{
            background-color: {theme["secondary_background"]} !important;
        }}
        [data-testid="stAlert"][data-testid="stAlert"],
        [data-testid="stAlert"][data-testid="stAlert"] * {{
            color: {theme["text"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Analytics logic
@st.cache_data(show_spinner="Computing t-SNE projection...")
def compute_tsne_coords(embeddings: list) -> list:
    """
    Reduce CLIP embedding vectors (usually 512-dim) down to 2D points so they
    can be plotted on a scatter chart — this is what lets you visually see
    which images CLIP considers "similar" (they'll cluster together).

    @st.cache_data means this only re-runs when the embeddings actually
    change (i.e. after re-indexing) — not on every widget interaction —
    since t-SNE is too slow to recompute on every Streamlit re-run.

    Takes in:
        embeddings: list of embedding vectors (one per indexed image)

    Requires scikit-learn (`pip install scikit-learn`), which isn't a
    dependency of the rest of the app, so this is imported lazily here
    rather than at the top of the file — that way the app still runs fine
    without it unless you actually open the Analytics tab.
    """
    import numpy as np
    from sklearn.manifold import TSNE

    vectors = np.array(embeddings)

    # Perplexity must be less than the number of samples — default of 30
    # breaks on small collections, so scale it down for tiny datasets
    perplexity = max(2, min(30, len(vectors) - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    return tsne.fit_transform(vectors).tolist()


# Indexing logic
def index_images(data_dir: str, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    Walk a folder, embed every image, and store vectors in ChromaDB
    (also displays a Streamlit progress bar during indexing)

    Takes in:

        data_dir:  Path to the folder of images to index

        embedder:  Loaded CLIPEmbedder instance

        indexer:   Connected VectorIndexer instance
    """

    # basically almost a copy-paste from main.py logic wise
    # in a professional project, main.py & app.py would import the functions from a shared pipeline file
    data_path = Path(data_dir)

    if not data_path.exists():
        st.error(f"Folder not found: `{data_dir}`. Create it and add images first.")
        return

    image_paths = [
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        st.warning(f"No supported images found in `{data_dir}`.")
        return

    # Fetch all existing IDs from ChromaDB ONCE before the loop:
    # Important for efficiency: querying the DB inside the loop would mean 1 DB round-trip per image (slow) 

    # Fetching once and storing as a Python set means each check is an O(1) hash lookup in memory (instant)
    existing_ids = set(indexer.collection.get()["ids"])

    # Progress bar widget (update it inside loop with float from 0.0 → 1.0)
    progress_bar = st.progress(0, text="Starting indexing...")
    status       = st.empty()  # placeholder

    embeddings, paths_str, failed = [], [], []
    skipped = 0

    for i, image_path in enumerate(image_paths):
        try:

            # Check if this image is already in the index by comparing its hash against set of existing IDs fetched above
            # If it's already there, update the progress bar and skip it (no need to re-embed something that hasntt changed)
            if indexer._path_to_id(str(image_path)) in existing_ids:
                skipped += 1
                progress = (i + 1) / len(image_paths)   
                progress_bar.progress(
                    progress,
                    text=f"Skipping {i + 1}/{len(image_paths)}: {image_path.name} (already indexed)"
                )
                continue

            # Call embedder.embed_image() & append to embeddings and paths_str
            embedding = embedder.embed_image(str(image_path))
            embeddings.append(embedding)
            paths_str.append(str(image_path))

            # Update progress bar (value must be a float between 0.0 & 1.0)
            progress = (i + 1) / len(image_paths)
            progress_bar.progress(progress, text=f"Embedding {i + 1}/{len(image_paths)}: {image_path.name}")

        except Exception as e:
            failed.append(image_path.name)
            status.warning(f"Skipped `{image_path.name}`: {e}")

    if embeddings:
        metadatas = [{"filename": Path(p).name} for p in paths_str]

        # Now call indexer.add_batch() with embeddings, paths_str, metadatas
        indexer.add_batch(embeddings, paths_str, metadatas)

    progress_bar.empty()  # Remove the progress bar once done

    st.success(
        f"Indexed {len(embeddings)} new images. "
        f"{skipped} already indexed (skipped). "
        f"{len(failed)} failed. "
        f"Total in index: {indexer.count()}"
    )


# Search logic
def run_search(query: str, top_k: int, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    Embed a text query, search ChromaDB, and render results as an image grid

    Takes in:

        query:    Natural language search string

        top_k:    Number of results to display (5)

        embedder: Loaded CLIPEmbedder instance

        indexer:  Connected VectorIndexer instance
    """

    # very similar to the search function in main.py
    if indexer.count() == 0:
        st.warning("Your index is empty. Go to the **Index Images** tab and index a folder first.")
        return

    # wrap the search in with st.spinner("Searching..."):
    # tells website to show a loading animation so user knows app didnt crash
    with st.spinner("Searching..."):
        
        # Embed query & search the index
        query_vector = embedder.embed_text(query)
        results      = indexer.search(query_vector, top_k=top_k)

    if not results:
        st.info("No results found. Try a different query.")
        return

    st.markdown(f"**{len(results)} results** for *\"{query}\"*")


    # Render and display results as a responsive image grid:
    # st.columns(n) splits the layout into n equal columns

    # 3 columns so results display as a grid rather than a vertical list
    # zip() pairs each result with a column (when results run out zip stops)
    cols = st.columns(3)

    st.write("")  # small breathing room before the grid

    _filename_theme = DARK_THEME if st.session_state.get("dark_mode", False) else LIGHT_THEME

    for i, result in enumerate(results):
        col = cols[i % 3]  # cycle through columns: 0, 1, 2, 0, 1, 2 ...

        with col:
            # st.container(border=True) is a built-in Streamlit primitive that draws
            # a subtle card outline — gives each result a "card" feel with no custom CSS
            with st.container(border=True):
                image_path = result["image_path"]

                if os.path.exists(image_path):
                    image = Image.open(image_path) 

                    # st.image() renders a PIL Image in the UI
                    # `use_container_width=True` makes it fill the column width responsively
                    st.image(image, width='stretch') # display image
                else:
                    st.warning(f"Image not found on disk: `{image_path}`")

                # Display filename and similarity score below each image (similar in main.py)
                similarity = 1 - result["distance"]
                filename   = result["metadata"].get("filename", Path(image_path).name)

                # Rendered as themed markdown (not st.caption()) so the filename
                # label follows dark/light mode instead of staying Streamlit's
                # fixed caption gray, same fix as the other labels in this file.
                st.markdown(
                    f'<p style="color:{_filename_theme["text"]}; margin:0;"><strong>{filename}</strong></p>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(similarity, 0.0), 1.0), text=f"Similarity: {similarity:.3f}")



# Page config & layout
# st.set_page_config() must be the 1st Streamlit call in the script 
# (tells the browser how to render the app before any other content appears)

# Controls the browser tab title, icon, and layout width
# `layout="wide"` uses full browser width instead of a narrow centered column
st.set_page_config(
    page_title="Image Search",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Local Image Search")
# st.caption() bakes in its own fixed low-opacity gray, same issue as the
# sidebar "Connected —" status — rendered as themed markdown instead so it
# follows dark/light mode. Read from session_state (not the `dark_mode`
# variable) since the sidebar toggle hasn't run yet at this point in the script.
# No opacity dimming here (unlike an earlier version of this fix) so the
# color swap reads as a true black/white shift, matching the title above it.
_subtitle_theme = DARK_THEME if st.session_state.get("dark_mode", False) else LIGHT_THEME
st.markdown(
    f'<p style="color:{_subtitle_theme["text"]}; margin:0;">'
    f'Semantic image search powered by CLIP + ChromaDB (runs entirely on your machine)</p>',
    unsafe_allow_html=True,
)
st.write("")  # extra whitespace under the header for a cleaner, less cramped feel


# Sidebar settings:
# st.sidebar is a special container that renders a collapsible panel on the left side of the page 
# (Good for settings that dont belong in the main content area)
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("")

    st.subheader("Appearance")
    # st.session_state already holds the post-click value by the time the
    # script reruns, so we can read it here to pick the label before the
    # widget is (re)created with that same value.
    theme_label = "🌙 Dark mode" if st.session_state.get("dark_mode", False) else "☀️ Light mode"
    dark_mode = st.toggle(theme_label, key="dark_mode")
    apply_theme(dark_mode)

    st.write("")
    st.subheader("Storage")
    index_dir = st.text_input(
        "Index directory",
        value=DEFAULT_INDEX_DIR,
    )

    data_dir = st.text_input(
        "Image folder",
        value=DEFAULT_DATA_DIR,
    )

    st.write("")
    st.subheader("Search")
    top_k = st.slider(
        "Results to show",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.divider()

    # Show current index size so the user knows if indexing has run
    # Wrap this in a try/except because load_indexer() may fail if the index_dir path is invalid (dont want the sidebar to crash)
    try:
        _indexer = load_indexer(index_dir)
        # st.caption() bakes in its own fixed low-opacity gray text color,
        # which our theme CSS override can't reliably win against — so this
        # message is rendered as styled markdown instead, using the active
        # theme's text color directly, with no opacity dimming so it reads
        # as true black/white like the title.
        _status_theme = DARK_THEME if dark_mode else LIGHT_THEME
        st.markdown(
            f'<p style="color:{_status_theme["text"]}; margin:0;">'
            f'✅ Connected — {_indexer.count()} vectors in index</p>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Could not connect to index: {e}")


# Quick stats dashboard: gives the page a "home screen" feel right under the title
# instead of jumping straight into tabs — this is the kind of thing that actually
# reads as "modern" (glanceable numbers up top) vs. a plain settings-and-form page
try:
    _dashboard_indexer = load_indexer(index_dir)
    _total_vectors = _dashboard_indexer.count()
except Exception:
    _total_vectors = 0

stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    with st.container(border=True):
        st.metric("🖼️ Images Indexed", _total_vectors)
with stat_col2:
    with st.container(border=True):
        st.metric("📁 Index Location", index_dir)
with stat_col3:
    with st.container(border=True):
        st.metric("🧠 Model", "CLIP")

st.write("")

# Main tabs:
# st.tabs() creates a tabbed layout: each `with tab:` block renders content only when that tab is active
# (keeps UI clean by separating indexing workflow from search workflow)
tab_search, tab_index, tab_analytics = st.tabs(["🔎  Search", "📥  Index Images", "📊  Analytics"])

# --- Search tab ---
with tab_search:
    st.write("")

    # Put the input and button on one row so "search" reads as a single, clean action
    # rather than a stacked form — a small but noticeable modernization
    col_query, col_button = st.columns([5, 1], vertical_alignment="bottom")

    with col_query:
        query = st.text_input(
            "Search your images",
            placeholder="ex: a photo of a cat",
            label_visibility="collapsed",
        )

    with col_button:
        search_clicked = st.button("Search", type="primary", width='stretch')

    st.write("")

    # st.button() returns True only on the re-run triggered by a click

    # Also trigger search if user presses Enter in the text input
    # by checking if `query` is non-empty and the button was clicked
    if search_clicked and query.strip():
        embedder = load_embedder()
        indexer  = load_indexer(index_dir)
        run_search(query.strip(), top_k, embedder, indexer)
    elif not query.strip():
        st.info("💡 Enter a text query above and click **Search** to get started.")

# --- Index tab ---
with tab_index:
    st.write("")
    with st.container(border=True):
        _index_theme = DARK_THEME if dark_mode else LIGHT_THEME

        # Split into a text column and an action column so "Start Indexing"
        # reads as a clear call-to-action next to the info, instead of just
        # sitting stacked underneath a wall of text
        col_info, col_action = st.columns([3, 1], vertical_alignment="center")

        with col_info:
            st.markdown("#### 📥 Index Your Images")
            st.markdown(
                f"Index all images from **`{data_dir}`** into ChromaDB.  \n"
                "Re-indexing is safe — existing images are updated, not duplicated."
            )

            # Quick preview of how many images are sitting in the folder, so the user
            # knows what "Start Indexing" is about to do before they click it
            # (rendered as themed markdown, not st.caption(), for the same
            # gray-that-never-changes reason as the other captions above — no
            # opacity dimming, so it swaps between true black/white like the title)
            # Shown as a small rounded "badge" (tinted with the theme's primary
            # color) instead of a plain line of text, so it reads as a status
            # chip rather than another paragraph.
            _preview_path = Path(data_dir)
            if _preview_path.exists():
                _preview_count = len([
                    p for p in _preview_path.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                ])
                st.markdown(
                    f'<div style="display:inline-block; margin:0.6rem 0 0.4rem 0; padding:0.3rem 0.9rem; '
                    f'border-radius:999px; background-color:{_index_theme["primary"]}26; '
                    f'color:{_index_theme["text"]}; font-size:0.85rem; font-weight:600;">'
                    f'📸 {_preview_count} supported image(s) found in this folder</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:inline-block; margin:0.6rem 0 0.4rem 0; padding:0.3rem 0.9rem; '
                    f'border-radius:999px; background-color:{_index_theme["primary"]}26; '
                    f'color:{_index_theme["text"]}; font-size:0.85rem; font-weight:600;">'
                    f'⚠️ This folder doesn\'t exist yet</div>',
                    unsafe_allow_html=True,
                )
            # A little breathing room so the badge above doesn't sit flush
            # against the card's bottom border
            st.write("")

        with col_action:
            if st.button("Start Indexing", type="primary", width='stretch'):
                embedder = load_embedder()
                indexer  = load_indexer(index_dir)
                index_images(data_dir, embedder, indexer)

                # st.rerun() forces Streamlit to re-run the script immediately
                # Used here so the sidebar metric updates right after indexing completes
                st.rerun()  

# --- Analytics tab ---
with tab_analytics:
    st.write("")
    _analytics_theme = DARK_THEME if dark_mode else LIGHT_THEME

    try:
        _analytics_indexer = load_indexer(index_dir)
        # collection.get() is the same public chromadb collection object
        # already used elsewhere in this file (see index_images()) — no
        # changes needed to indexer.py to support this tab. `include` must
        # be set explicitly since chromadb doesn't return embeddings by
        # default (they're the expensive part of the payload).
        _all_data = _analytics_indexer.collection.get(include=["embeddings", "metadatas"])
    except Exception as e:
        st.error(f"Could not load index: {e}")
        _all_data = {"ids": [], "embeddings": [], "metadatas": []}

    _ids = _all_data.get("ids", [])

    if len(_ids) == 0:
        with st.container(border=True):
            st.markdown("#### 📊 Image Analytics")
            st.markdown(
                "No images indexed yet — head to the **Index Images** tab first, "
                "then come back here to see your collection's stats."
            )
    else:
        _metadatas  = _all_data.get("metadatas", [])
        _embeddings = _all_data.get("embeddings", [])

        # --- Stat cards: total indexed + total storage on disk ---
        # Storage is measured from the actual files in data_dir (not the
        # vectors themselves, which are tiny) so it reflects the real
        # disk footprint of the image collection being indexed
        _total_bytes = 0
        _data_path = Path(data_dir)
        if _data_path.exists():
            for p in _data_path.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    _total_bytes += p.stat().st_size
        _total_mb = _total_bytes / (1024 * 1024)

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            with st.container(border=True):
                st.metric("🗂️ Images Indexed", len(_ids))
        with stat_col2:
            with st.container(border=True):
                st.metric("💾 Storage Used", f"{_total_mb:.1f} MB")

        st.write("")

        # Real indexer.py only guarantees an "image_path" key in metadata
        # (added automatically by add()/add_batch()) — "filename" is an
        # extra key app.py's own index_images() happens to pass in, so it
        # won't be there for anything indexed another way (e.g. via
        # main.py, or a direct indexer.add() call). Fall back to deriving
        # it from image_path so this tab doesn't blow up on those records.
        def _get_filename(meta: dict) -> str:
            if meta.get("filename"):
                return meta["filename"]
            return Path(meta.get("image_path", "unknown")).name

        # --- File extension breakdown (bar chart) ---
        with st.container(border=True):
            st.markdown("#### 🗃️ File Types")
            _ext_counts: dict = {}
            for meta in _metadatas:
                _ext = Path(_get_filename(meta)).suffix.lower() or "unknown"
                _ext_counts[_ext] = _ext_counts.get(_ext, 0) + 1

            _ext_df = pd.DataFrame(
                {"extension": list(_ext_counts.keys()), "count": list(_ext_counts.values())}
            )
            _ext_chart = (
                alt.Chart(_ext_df)
                .mark_bar(color=_analytics_theme["primary"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("extension", title=None, axis=alt.Axis(labelColor=_analytics_theme["text"])),
                    y=alt.Y("count", title="Images", axis=alt.Axis(labelColor=_analytics_theme["text"])),
                    tooltip=["extension", "count"],
                )
                .properties(height=260)
                .configure_axis(gridColor=_analytics_theme["border"] + "22")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(_ext_chart, use_container_width=True)

        st.write("")

        # --- t-SNE 2D projection of the CLIP embeddings ---
        with st.container(border=True):
            st.markdown("#### 🧠 How CLIP Sees Your Images")
            st.markdown(
                "Each image's 512-dimensional CLIP embedding is projected down to 2D. "
                "Images CLIP considers visually/semantically similar will cluster closer together."
            )

            if len(_ids) < 4:
                st.markdown(
                    f'<div style="display:inline-block; margin:0.6rem 0 0.4rem 0; padding:0.3rem 0.9rem; '
                    f'border-radius:999px; background-color:{_analytics_theme["primary"]}26; '
                    f'color:{_analytics_theme["text"]}; font-size:0.85rem; font-weight:600;">'
                    f'Index at least 4 images to see the 2D projection</div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    _coords = compute_tsne_coords(_embeddings)
                    _filenames = [_get_filename(m) for m in _metadatas]
                    _exts = [Path(f).suffix.lower() or "unknown" for f in _filenames]

                    _proj_df = pd.DataFrame({
                        "x": [c[0] for c in _coords],
                        "y": [c[1] for c in _coords],
                        "filename": _filenames,
                        "type": _exts,
                    })

                    _scatter = (
                        alt.Chart(_proj_df)
                        .mark_circle(size=110, opacity=0.85)
                        .encode(
                            x=alt.X("x", axis=None),
                            y=alt.Y("y", axis=None),
                            color=alt.Color(
                                "type",
                                legend=alt.Legend(title="File type", labelColor=_analytics_theme["text"], titleColor=_analytics_theme["text"]),
                            ),
                            tooltip=["filename", "type"],
                        )
                        .properties(height=420)
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(_scatter, use_container_width=True)
                except ModuleNotFoundError:
                    st.error(
                        "This needs scikit-learn, which isn't installed. Run "
                        "`pip install scikit-learn` (add `--break-system-packages` if needed), "
                        "then reload this tab."
                    )