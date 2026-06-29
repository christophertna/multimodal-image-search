"""
main.py is the CLI entry point that wires embedder.py & indexer.py together

Usage:
    # Index all images in a folder:
    python main.py --mode index --data_dir ./data/images

    # Search with a text query:
    python main.py --mode search --query "a dog on a beach"

    # Search and return more results:
    python main.py --mode search --query "sunset over mountains" --top_k 10

Architecture note:

    main.py owns 1 responsibility: orchestration
    
    Doesnt know how embeddings are generated (embedder.py's job),
    nor how vectors are stored (indexer.py's job)

    Just calls both in the right order and presents the results

    Separation means we can swap out model or db without touching main.py
"""

import argparse
import os
from pathlib import Path

# recall from __init__.py, import these
from src.embedder import CLIPEmbedder
from src.indexer import VectorIndexer


# Constants
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} # Supported image extensions 
# Path.suffix returns e.g. ".jpg" (lowercase)

DEFAULT_INDEX_DIR  = "./index"
DEFAULT_TOP_K      = 5


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
        choices=["index", "search"],    # `choices` restricts the value to a fixed set: argparse errors automatically if user passes anything else
        required=True,                  # `required=True` means flag must always be provided
        help="'index' to embed and store images. 'search' to query by text.",
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

    # Initialize shared components
    # Both modes need the embedder and indexer, so we initialize them once here and pass them into whichever mode function runs

    # This avoids loading the CLIP model twice if we ever chain modes
    print("[main] Loading CLIP model...")
    embedder = CLIPEmbedder()

    print("[main] Connecting to ChromaDB...")
    indexer = VectorIndexer(persist_dir=args.index_dir)

    # Dispatch to the correct mode
    # For 2 modes, if/elif is clearer but for like 5+ modes, dict approach scales better
    if args.mode == "index":
        run_index(args.data_dir, embedder, indexer)
    elif args.mode == "search":
        run_search(args.query, args.top_k, embedder, indexer)

# This guard means the code inside only runs when you execute this file directly (`python main.py`) 
# If another file imports main.py, the block is skipped (preventing unintended side effects on import)
if __name__ == "__main__":
    main()