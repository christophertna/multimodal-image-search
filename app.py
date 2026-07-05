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
    "text": "#2B2B2B",
    "primary": "#5B6C8F",
    "input_background": "#FFFFFF",
    "input_text": "#2B2B2B",
}
DARK_THEME = {
    "background": "#1E1E1E",
    "secondary_background": "#2A2A2A",
    "text": "#EAEAEA",
    "primary": "#7C8DB5",
    "input_background": "#333333",
    "input_text": "#F5F5F5",
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
    Inject a small CSS override so the sidebar toggle can switch colors live.

    This is the one place custom CSS is used in the whole file — everything
    else relies on Streamlit's built-in theming/layout. Targets the actual
    app/sidebar/button elements directly instead of guessing at Streamlit's
    internal CSS variable names, so it's more likely to hold up across versions.
    """
    theme = DARK_THEME if dark else LIGHT_THEME

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}
        /* The top toolbar strip Streamlit renders above the page content —
           left un-styled it stays white/default even when the rest of the
           app is dark, which is the "white band" above the title. */
        [data-testid="stHeader"] {{
            background-color: {theme["background"]} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {theme["secondary_background"]} !important;
        }}
        /* Streamlit sets its own inline text color on nested elements
           (captions, labels, metric text, etc.), not just the container,
           so a rule on the container alone doesn't reach them. Some of
           those inline colors come from a selector that's genuinely MORE
           specific than a normal wildcard, so a plain "* {{ !important }}"
           can still lose that tie. Repeating the same attribute selector
           several times in a row (e.g. [data-testid="x"][data-testid="x"])
           matches the exact same element but stacks extra specificity
           weight on our side without needing to guess Streamlit's internal
           class names — that's the trick used everywhere below. */
        [data-testid="stSidebar"][data-testid="stSidebar"][data-testid="stSidebar"] * {{
            color: {theme["text"]} !important;
        }}
        /* Text inputs get their own background + text pair per mode —
           they don't need to match the page colors, just be readable.
           This selector is more specific than the wildcard above, so it
           still wins for the actual input text. Placeholder text (the
           greyed-out example shown before you type) is styled through the
           separate ::placeholder pseudo-element, so it needs its own rule
           or it stays whatever gray the browser/Streamlit defaults to. */
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
        /* st.container(border=True) cards keep their own background AND
           their own inline text colors on nested elements by default —
           this is what was making "Images Indexed" / "Index Location" /
           "Model" hard to read even after the container background fix. */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {theme["secondary_background"]} !important;
        }}
        /* The metric label/value and caption text ("Images Indexed",
           "Connected — x vectors in index", etc.) get their color set by
           Streamlit directly on the innermost text element using a more
           specific selector than a plain wildcard — same specificity-
           stacking trick as the sidebar rule above, applied to every
           plausible element/testid combination so it holds regardless of
           which exact one Streamlit uses under the hood. */
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
        /* Tab labels ("Search" / "Index Images") get their text color inline
           from Streamlit's own theme engine too — same story, same fix. */
        body .stTabs [data-baseweb="tab"],
        body .stTabs [data-baseweb="tab"] * {{
            color: {theme["text"]} !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {theme["primary"]} !important;
            border-color: {theme["primary"]} !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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

                # st.caption() renders small grey text (for the labels under images)
                st.caption(f"**{filename}**")
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
st.caption("Semantic image search powered by CLIP + ChromaDB (runs entirely on your machine)")
st.write("")  # extra whitespace under the header for a cleaner, less cramped feel


# Sidebar settings:
# st.sidebar is a special container that renders a collapsible panel on the left side of the page 
# (Good for settings that dont belong in the main content area)
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("")

    st.subheader("Appearance")
    dark_mode = st.toggle("🌙 Dark mode", key="dark_mode")
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
        st.caption(f"✅ Connected — {_indexer.count()} vectors in index")
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
tab_search, tab_index = st.tabs(["🔎  Search", "📥  Index Images"])

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
        st.markdown(
            f"Index all images from **`{data_dir}`** into ChromaDB.  \n"
            "Re-indexing is safe — existing images are updated, not duplicated."
        )

        # Quick preview of how many images are sitting in the folder, so the user
        # knows what "Start Indexing" is about to do before they click it
        _preview_path = Path(data_dir)
        if _preview_path.exists():
            _preview_count = len([
                p for p in _preview_path.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ])
            st.caption(f"📸 {_preview_count} supported image(s) found in this folder.")
        else:
            st.caption("⚠️ This folder doesn't exist yet.")

        if st.button("Start Indexing", type="primary"):
            embedder = load_embedder()
            indexer  = load_indexer(index_dir)
            index_images(data_dir, embedder, indexer)

            # st.rerun() forces Streamlit to re-run the script immediately
            # Used here so the sidebar metric updates right after indexing completes
            st.rerun()