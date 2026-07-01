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

    for i, result in enumerate(results):
        col = cols[i % 3]  # cycle through columns: 0, 1, 2, 0, 1, 2 ...

        with col:
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
            st.caption(f"**{filename}**  \nSimilarity: `{similarity:.3f}`")



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


# Sidebar settings:
# st.sidebar is a special container that renders a collapsible panel on the left side of the page 
# (Good for settings that dont belong in the main content area)
with st.sidebar:
    st.header("Settings")

    index_dir = st.text_input(
        "Index directory",
        value=DEFAULT_INDEX_DIR,
        help="Where ChromaDB stores its data on disk.",
    )

    data_dir = st.text_input(
        "Image folder",
        value=DEFAULT_DATA_DIR,
        help="Folder of images to index. Supports jpg, png, bmp, webp.",
    )

    top_k = st.slider(
        "Results to show",
        min_value=1,
        max_value=20,
        value=5,
        help="How many images to return per search.",
    )

    st.divider()

    # Show current index size so the user knows if indexing has run
    # Wrap this in a try/except because load_indexer() may fail if the index_dir path is invalid (dont want the sidebar to crash)
    try:
        _indexer = load_indexer(index_dir)
        st.metric("Vectors in index", _indexer.count())
    except Exception as e:
        st.error(f"Could not connect to index: {e}")


# Main tabs:
# st.tabs() creates a tabbed layout: each `with tab:` block renders content only when that tab is active
# (keeps UI clean by separating indexing workflow from search workflow)
tab_search, tab_index = st.tabs(["Search", "Index Images"])

# --- Search tab ---
with tab_search:
    query = st.text_input(
        "Search your images",
        placeholder="e.g. a dog on a beach, sunset over mountains, red car...",
    )

    # st.button() returns True only on the re-run triggered by a click

    # Also trigger search if user presses Enter in the text input
    # by checking if `query` is non-empty and the button was clicked
    if st.button("Search", type="primary") and query.strip():
        embedder = load_embedder()
        indexer  = load_indexer(index_dir)
        run_search(query.strip(), top_k, embedder, indexer)
    elif not query.strip():
        st.info("Enter a text query above and click Search.")

# --- Index tab ---
with tab_index:
    st.markdown(
        f"Index all images from **`{data_dir}`** into ChromaDB. "
        "Re-indexing is safe (existing images are updated, not duplicated)"
    )

    if st.button("Start Indexing", type="primary"):
        embedder = load_embedder()
        indexer  = load_indexer(index_dir)
        index_images(data_dir, embedder, indexer)

        # st.rerun() forces Streamlit to re-run the script immediately
        # Used here so the sidebar metric updates right after indexing completes
        st.rerun()