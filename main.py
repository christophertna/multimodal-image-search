"""
main.py is the CLI entry point that wires embedder.py & indexer.py together

Usage:
    # Index all images in a folder:
    python main.py --mode index --data_dir ./data/images

    # Search with a text query:
    python main.py --mode search --query "a dog on a beach"

    # Search and return more results:
    python main.py --mode search --query "sunset over mountains" --top_k 10

    # Reverse image search: find images similar to a QUERY IMAGE:
    python main.py --mode reverse_search --image_path ./query.jpg

    # Surprise me: search using a randomly picked already-indexed image:
    python main.py --mode surprise

    # Wipe the entire index:
    python main.py --mode reset

    # Vector metadata analysis (semantic range, redundant pairs, unique
    # images, density hotspots, cross-modal bridge) as text output:
    python main.py --mode analyze

Architecture note:

    main.py owns 1 responsibility: orchestration
    
    Doesnt know how embeddings are generated (embedder.py's job),
    nor how vectors are stored (indexer.py's job)

    Just calls both in the right order and presents the results

    Separation means we can swap out model or db without touching main.py
"""

import argparse
from pathlib import Path

# recall from __init__.py, import these
from src.embedder import CLIPEmbedder
from src.indexer import VectorIndexer


# Constants
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} # Supported image extensions 
# Path.suffix returns e.g. ".jpg" (lowercase)

DEFAULT_INDEX_DIR  = "./index"
DEFAULT_TOP_K      = 5

# Same generic category prompts as app.py's Analytics tab, used by the
# "Cross-Modal Bridge" analysis (--mode analyze) to test how well CLIP can
# describe each indexed image with a common, everyday label.
CROSS_MODAL_PROMPTS = (
    "a photo of a person",
    "a photo of a group of people",
    "a photo of an animal",
    "a photo of food",
    "a photo of a building or architecture",
    "a photo of nature or a landscape",
    "a photo of a vehicle",
    "a photo of technology or electronics",
    "a photo of text or a document",
    "a photo of art or a drawing",
    "a close-up photo of an object",
    "an abstract or pattern image",
)


# Index mode
def run_index(data_dir: str, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    Walk a folder, embed every image and store the vectors in ChromaDB

    Takes in:

        data_dir:  Path to the folder containing images to index

        embedder:  Initialized CLIPEmbedder instance

        indexer:   Initialized VectorIndexer instance
    """


    # Path validation check 
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Image directory not found: '{data_dir}'. "
            "Create the folder and add some images before indexing."
        )

    # Collect all valid image paths first so we can report the total count

    # A list comprehension with a condition to filter items & "Path.suffix" gives the file extension (.jpg, .png, etc.)
    # Build `image_paths`, a list of Path objects from data_path where the file suffix (lowercased) is in SUPPORTED_EXTENSIONS
    image_paths = [
        p for p in data_path.iterdir() # Path.iterdir() yields all files and folders directly inside a directory
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS # suffix matching loop
    ]

    # soft exit/return early so no program crashes
    if not image_paths:
        print(f"No supported images found in '{data_dir}'. "
              f"Supported formats: {SUPPORTED_EXTENSIONS}")
        return

    print(f"[Index] Found {len(image_paths)} images in '{data_dir}'. Starting indexing...")


    # Batch embedding + indexing
    # Collect all embeddings first, then insert as one batch

    # This is faster than calling indexer.add() inside the loop because ChromaDB does 1 disk write instead of N separate writes.
    embeddings = []
    paths_str  = []  # ChromaDB metadata expects plain strings not Path objects, so must convert later
    failed     = []

    for i, image_path in enumerate(image_paths):

        # try-except loop so even with 1 bad file, loop skips over it and continues
        try:

            # Call embedder.embed_image() with the image path as a string
            # Store the result by appending to `embeddings`
            embedding = embedder.embed_image(str(image_path))
            embeddings.append(embedding)
            paths_str.append(str(image_path))
            # ensure 'embeddings' list and 'paths_str' list have same length

            # Basic progress indicator: prints every 10 images
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths): # guarantee to print last image even if not modulo 10 = 0
                print(f"  Embedded {i + 1}/{len(image_paths)}: {image_path.name}")

        except Exception as e:

            # Skip over bad files instead of crashing & print warning message for that image
            print(f"  [WARN] Skipping '{image_path.name}': {e}")
            failed.append(str(image_path))

    # Build optional metadata for each image in case (as a list of dictionnaries)
    # Store the filename separately from the full path so its easier to filter/display later without parsing full path string
    metadatas = [{"filename": Path(p).name} for p in paths_str]

    # Call indexer.add_batch() with embeddings, paths_str, and metadatas
    ids = indexer.add_batch(embeddings, paths_str, metadatas) # add by batch for efficiency

    print(f"\n[Index] Done. {len(ids)} images indexed, {len(failed)} skipped.")
    print(f"[Index] Total vectors in collection: {indexer.count()}")

    if failed:
        print(f"[Index] Failed files: {failed}")



# Search mode
def run_search(query: str, top_k: int, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    Embed a text query and retrieve the most similar images from the index

    Takes in:

        query:    Natural language search string

        top_k:    Number of results to return (5)

        embedder: Initialized CLIPEmbedder instance

        indexer:  Initialized VectorIndexer instance
    """

    print(f"\n[Search] Query: '{query}'") #reprint query      
    print(f"[Search] Searching top {top_k} results...\n")

    # Embed the query string using embedder.embed_text() & store the result in `query_vector`
    query_vector = embedder.embed_text(query)

    # Call indexer.search() with query_vector & top_k then store returned list of result dicts in `results`
    results = indexer.search(query_vector, top_k=top_k) # recall search method in indexer.py

    if not results:
        print("[Search] No results found. Try a different query or re-index your images.")
        return

    # Display results:
    # enumerate(iterable, start=1) gives (1, item), (2, item), etc, so we get a 1-based rank counter 
    for rank, result in enumerate(results, start=1):

        # Cosine distance from ChromaDB is in range [0, 2] where: 0 = identical vectors, 2 = completely opposite
        # Converting to similarity (1 - distance) to make it cleaner where: 1.0 = perfect match and 0.0 = no similarity
        similarity = 1 - result["distance"]

        print(f"  Rank {rank}:")
        print(f"    Path:       {result['image_path']}")
        print(f"    Similarity: {similarity:.4f}")
        print(f"    Metadata:   {result['metadata']}")
        print()


def _print_results(results: list) -> None:
    """
    Shared result-printing logic — factored out so run_search(),
    run_reverse_search(), and run_surprise() don't each duplicate the same
    rank/path/similarity/metadata print block (mirrors app.py's own
    _render_search_results(), which exists for the exact same reason).
    """
    for rank, result in enumerate(results, start=1):
        similarity = 1 - result["distance"]
        print(f"  Rank {rank}:")
        print(f"    Path:       {result['image_path']}")
        print(f"    Similarity: {similarity:.4f}")
        print(f"    Metadata:   {result['metadata']}")
        print()


# Reverse image search mode
def run_reverse_search(image_path: str, top_k: int, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    Embed a QUERY IMAGE (instead of a text string) and retrieve the most
    similar images from the index — CLI equivalent of app.py's reverse
    image search mode.

    Takes in:

        image_path: Path to the query image on disk

        top_k:      Number of results to return

        embedder:   Initialized CLIPEmbedder instance

        indexer:    Initialized VectorIndexer instance
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Query image not found: '{image_path}'")

    print(f"\n[Reverse Search] Query image: '{image_path}'")
    print(f"[Reverse Search] Searching top {top_k} results...\n")

    # embedder.embed_image() only accepts a file path already sitting on
    # disk (see embedder.py) — unlike app.py, which has to write an
    # uploaded file to a temp path first, the CLI's --image_path IS
    # already a real path, so this can call it directly
    query_vector = embedder.embed_image(str(image_file))
    results = indexer.search(query_vector, top_k=top_k)

    if not results:
        print("[Reverse Search] No results found. Try a different image or re-index your images.")
        return

    _print_results(results)


# Surprise mode
def run_surprise(top_k: int, indexer: VectorIndexer) -> None:
    """
    Pick a random already-indexed image and search using its OWN stored
    embedding — CLI equivalent of app.py's "Surprise Me" button.

    Deliberately doesn't take an `embedder` argument at all: the randomly
    picked image's embedding is ALREADY sitting in ChromaDB from when it
    was indexed, so this reuses that exact vector rather than re-embedding
    anything — same reasoning as app.py's run_surprise_me().

    Takes in:

        top_k:   Number of results to return

        indexer: Initialized VectorIndexer instance
    """
    import random
    import numpy as np

    if indexer.count() == 0:
        print("[Surprise] Index is empty. Run --mode index first.")
        return

    all_data = indexer.collection.get(include=["embeddings", "metadatas"])
    ids = all_data["ids"]

    random_position = random.randrange(len(ids))
    # collection.get() hands back plain Python lists, which don't have the
    # .ndim/.shape/.tolist() attributes indexer.py's _validate_embedding()
    # expects — wrapping in numpy satisfies that duck typing without
    # needing to reconstruct a torch.Tensor (same fix as app.py's version)
    query_vector = np.array(all_data["embeddings"][random_position])
    picked_meta = all_data["metadatas"][random_position]
    picked_filename = picked_meta.get("filename") or Path(picked_meta.get("image_path", "?")).name

    print(f"\n[Surprise] Randomly picked: '{picked_filename}'")
    print(f"[Surprise] Searching top {top_k} similar images...\n")

    results = indexer.search(query_vector, top_k=top_k)
    _print_results(results)


# Reset mode
def run_reset(indexer: VectorIndexer) -> None:
    """
    Wipe the entire collection — CLI equivalent of app.py's "Reset Index"
    button.

    Not gated behind an interactive confirmation prompt the way the
    Streamlit UI's checkbox is: this is a scriptable CLI tool, and running
    `--mode reset` explicitly IS the confirmation. (A future --yes/--force
    flag could add an interactive y/n prompt instead, if that's ever
    wanted for safety.)

    Takes in:

        indexer: Initialized VectorIndexer instance
    """
    count_before = indexer.count()
    indexer.reset()
    print(f"[Reset] Collection wiped. {count_before} vector(s) removed.")


# Analyze mode — text port of app.py's "Vector Metadata Analysis" section.
# The File Types bar chart and t-SNE 2D projection aren't included here —
# neither has a meaningful plain-text form, so those stay Streamlit-only.
def _get_filename(meta: dict) -> str:
    """
    Same fallback logic as app.py's own _get_filename(): indexer.py only
    guarantees an "image_path" key in metadata (added automatically by
    add()/add_batch()) — "filename" is an extra key both app.py and
    main.py happen to add during indexing, so this stays safe for
    anything indexed another way.
    """
    if meta.get("filename"):
        return meta["filename"]
    return Path(meta.get("image_path", "unknown")).name


def compute_similarity_analysis(embeddings: list, top_n: int = 3, density_percentile: float = 99.0) -> dict:
    """
    Same pairwise cosine similarity sweep as app.py's Analytics tab,
    covering Most Redundant Pairs, Most Unique Images, Collection Density
    Hotspots, and Semantic Range — ported here rather than imported, since
    main.py and app.py don't share a common module (see this file's own
    architecture note above about a possible future shared pipeline
    file). No @st.cache_data equivalent needed: this is a one-shot CLI
    run, not a Streamlit rerun loop.

    Takes in:
        embeddings:         list of embedding vectors (one per indexed image)
        top_n:               how many top pairs / outliers / hotspots to return
        density_percentile: similarity percentile that counts as a "close
                            neighbor" for the density/hotspot count (99.0 =
                            top 1% most-similar pairs in THIS collection)
    """
    import numpy as np

    vectors = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    n = len(vectors)
    sim_matrix = vectors @ vectors.T

    # --- Most redundant pairs ---
    iu = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[iu]
    top_pair_positions = np.argsort(pair_sims)[::-1][:top_n]
    redundant_pairs = [
        (int(iu[0][pos]), int(iu[1][pos]), float(pair_sims[pos]))
        for pos in top_pair_positions
    ]

    # --- Collection density hotspots ---
    sim_matrix_no_diag = sim_matrix.copy()
    np.fill_diagonal(sim_matrix_no_diag, -1.0)
    density_threshold = float(np.percentile(pair_sims, density_percentile))
    neighbor_counts = (sim_matrix_no_diag >= density_threshold).sum(axis=1)
    top_density_positions = np.argsort(neighbor_counts)[::-1][:top_n]
    density_hotspots = [
        (int(pos), int(neighbor_counts[pos])) for pos in top_density_positions
    ]

    # --- Semantic range ---
    centroid = vectors.mean(axis=0)
    centroid_magnitude = float(np.linalg.norm(centroid))
    if centroid_magnitude > 0:
        centroid_unit = centroid / centroid_magnitude
        similarity_to_centroid = vectors @ centroid_unit
        distances_from_centroid = 1.0 - similarity_to_centroid
    else:
        distances_from_centroid = np.ones(n)

    semantic_range = {
        "avg_distance": float(distances_from_centroid.mean()),
        "std_distance": float(distances_from_centroid.std()),
        "centroid_coherence": centroid_magnitude,
    }

    # --- Most unique images ---
    np.fill_diagonal(sim_matrix, 0.0)
    avg_similarity = sim_matrix.sum(axis=1) / max(n - 1, 1)
    most_unique_positions = np.argsort(avg_similarity)[:top_n]
    unique_images = [
        (int(pos), float(avg_similarity[pos])) for pos in most_unique_positions
    ]

    return {
        "redundant_pairs": redundant_pairs,
        "unique_images": unique_images,
        "density_hotspots": density_hotspots,
        "density_threshold": density_threshold,
        "semantic_range": semantic_range,
    }


def compute_cross_modal_scores(
    embeddings: list, prompts: tuple, embedder: CLIPEmbedder, top_n: int = 3
) -> list:
    """
    Same "Cross-Modal Bridge" analysis as app.py's Analytics tab — finds
    each image's single best-matching prompt, then returns the images
    whose best match is still weakest (i.e. hardest for CLIP to
    confidently describe with any common label).

    Takes embedder as an explicit parameter rather than fetching it via a
    cache (main.py has no Streamlit-style resource caching) — main()
    already loads it once and passes it around like every other mode here does.
    """
    import numpy as np

    image_vectors = np.array(embeddings, dtype=np.float64)
    image_norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
    image_norms[image_norms == 0] = 1
    image_vectors = image_vectors / image_norms

    prompt_vectors = np.array([embedder.embed_text(p).detach().cpu().numpy() for p in prompts])
    prompt_norms = np.linalg.norm(prompt_vectors, axis=1, keepdims=True)
    prompt_norms[prompt_norms == 0] = 1
    prompt_vectors = prompt_vectors / prompt_norms

    sim_matrix = image_vectors @ prompt_vectors.T
    best_prompt_idx = sim_matrix.argmax(axis=1)
    best_scores = sim_matrix.max(axis=1)

    weakest_positions = np.argsort(best_scores)[:top_n]

    return [
        (int(pos), prompts[int(best_prompt_idx[pos])], float(best_scores[pos]))
        for pos in weakest_positions
    ]


def run_analyze(top_n: int, embedder: CLIPEmbedder, indexer: VectorIndexer) -> None:
    """
    CLI text port of app.py's "Vector Metadata Analysis" section:
    Semantic Range, Most Redundant Pairs, Most Unique Images, Collection
    Density Hotspots, and Cross-Modal Bridge.

    Takes in:
        top_n:    How many top pairs/images to show per section (reuses
                  --top_k for this, same "how many things to show" idea)
        embedder: Initialized CLIPEmbedder instance (needed for the
                  Cross-Modal Bridge section's text prompts)
        indexer:  Initialized VectorIndexer instance
    """
    # Same safety cap as app.py's Analytics tab — the O(n²) similarity
    # matrix is trivial for a personal collection of a few thousand
    # images, but shouldn't silently balloon in memory well beyond that
    ANALYSIS_CAP = 3000

    count = indexer.count()
    if count < 3:
        print(f"[Analyze] Need at least 3 indexed images to run this analysis (found {count}).")
        return
    if count > ANALYSIS_CAP:
        print(
            f"[Analyze] Skipped — over {ANALYSIS_CAP} images ({count} indexed). "
            f"A full pairwise sweep at this size would use significant memory."
        )
        return

    print(f"\n[Analyze] Running vector metadata analysis on {count} indexed images...\n")

    all_data = indexer.collection.get(include=["embeddings", "metadatas"])
    embeddings = all_data["embeddings"]
    metadatas = all_data["metadatas"]
    filenames = [_get_filename(m) for m in metadatas]

    sim_results = compute_similarity_analysis(embeddings, top_n=top_n)

    srange = sim_results["semantic_range"]
    print("Semantic Range")
    print("-" * 40)
    print(f"  Avg. distance from center: {srange['avg_distance']:.3f}")
    print(f"  Spread (std. dev):         {srange['std_distance']:.3f}")
    print(f"  Centroid coherence:        {srange['centroid_coherence']:.3f}")
    print()

    print("Most Redundant Pairs")
    print("-" * 40)
    if not sim_results["redundant_pairs"]:
        print("  Nothing to show.")
    for rank, (i, j, sim) in enumerate(sim_results["redundant_pairs"], start=1):
        print(f"  {rank}. {sim:.3f} similarity — {filenames[i]} <-> {filenames[j]}")
    print()

    print("Most Unique Images")
    print("-" * 40)
    if not sim_results["unique_images"]:
        print("  Nothing to show.")
    for rank, (idx, avg_sim) in enumerate(sim_results["unique_images"], start=1):
        print(f"  {rank}. {avg_sim:.3f} avg. similarity — {filenames[idx]}")
    print()

    print("Collection Hotspots")
    print("-" * 40)
    if not sim_results["density_hotspots"]:
        print("  Nothing to show.")
    for rank, (idx, neighbor_count) in enumerate(sim_results["density_hotspots"], start=1):
        plural = "s" if neighbor_count != 1 else ""
        print(f"  {rank}. {neighbor_count} close neighbor{plural} — {filenames[idx]}")
    print()

    print("Cross-Modal Bridge (Hard-to-Classify Images)")
    print("-" * 40)
    bridge_results = compute_cross_modal_scores(embeddings, CROSS_MODAL_PROMPTS, embedder, top_n=top_n)
    if not bridge_results:
        print("  Nothing to show.")
    for rank, (idx, best_prompt, best_score) in enumerate(bridge_results, start=1):
        print(f'  {rank}. {best_score:.3f} best match ("{best_prompt}") — {filenames[idx]}')
    print()


# CLI argument parsing
def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments:

    Why argparse?

        argparse is Python's built-in CLI argument library. It automatically generates --help output, validates types, 
        and handles missing required arguments with clear error messages (no 3rd party library needed)
    """

    # ArgumentParser is the main class and `description` appears in --help output
    parser = argparse.ArgumentParser(
        description="Local multimodal image search engine using CLIP + ChromaDB."
    )

    parser.add_argument(                # add_argument() defines each flag
        "--mode",
        type=str,
        choices=["index", "search", "reverse_search", "surprise", "reset", "analyze"],
        required=True,                  # `required=True` means flag must always be provided
        help=(
            "'index' to embed and store images. 'search' to query by text. "
            "'reverse_search' to query by image (needs --image_path). "
            "'surprise' to search using a randomly picked already-indexed image. "
            "'reset' to wipe the entire index."
        ),
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the image folder. Required when --mode index.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text search query. Required when --mode search.",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a query image. Required when --mode reverse_search.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return in search mode. Default: {DEFAULT_TOP_K}.",
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory where ChromaDB persists its data. Default: '{DEFAULT_INDEX_DIR}'.",
    )

    return parser.parse_args()


# Entry point
def main() -> None:
    args = parse_args()

    # Validate that the right flags are provided for each mode
    # need to manually check if the correct flags are present for each mode (argparse doesnt support natively)
    if args.mode == "index" and not args.data_dir:
        raise ValueError("--data_dir is required when running in index mode.") # --data_dir must be provided if index mode was chosen
    if args.mode == "search" and not args.query:
        raise ValueError("--query is required when running in search mode.")   # --query must be provided if search mode was chosen
    if args.mode == "reverse_search" and not args.image_path:
        raise ValueError("--image_path is required when running in reverse_search mode.")

    print("[main] Connecting to ChromaDB...")
    indexer = VectorIndexer(persist_dir=args.index_dir)

    # Only load the (expensive) CLIP model for modes that actually need to
    # embed something new. 'reset' and 'surprise' both work entirely off
    # vectors already sitting in ChromaDB — loading the model for them
    # would just cost several seconds of startup time for nothing, same
    # reasoning as app.py never loading its embedder for Reset Index or
    # Surprise Me either.
    embedder = None
    if args.mode in ("index", "search", "reverse_search", "analyze"):
        print("[main] Loading CLIP model...")
        embedder = CLIPEmbedder()

    # Dispatch to the correct mode
    # For 2 modes, if/elif is clearer but for like 5+ modes, dict approach scales better
    if args.mode == "index":
        run_index(args.data_dir, embedder, indexer)
    elif args.mode == "search":
        run_search(args.query, args.top_k, embedder, indexer)
    elif args.mode == "reverse_search":
        run_reverse_search(args.image_path, args.top_k, embedder, indexer)
    elif args.mode == "surprise":
        run_surprise(args.top_k, indexer)
    elif args.mode == "reset":
        run_reset(indexer)
    elif args.mode == "analyze":
        run_analyze(args.top_k, embedder, indexer)

# This guard means the code inside only runs when you execute this file directly (`python main.py`) 
# If another file imports main.py, the block is skipped (preventing unintended side effects on import)
if __name__ == "__main__":
    main()