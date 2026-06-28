# Multimodal Image Search Engine

A local-first, high-performance semantic search engine for images using CLIP and vector embeddings.

## Project Roadmap
-  **Step 1: Environment Setup** - Virtual environment configured.
    - Git repository initialized.
    - Project structure created (`/src`, `/data`, `/index`).
-  **Step 2: Model Integration**
    - `src/embedder.py` implemented using `CLIPModel` and `CLIPProcessor`.
    - Device-agnostic code (CUDA/MPS/CPU) and L2-normalization established.
-  **Step 3: Vector Indexing (In Progress)**
    - Implement `src/indexer.py` to batch-process images.
    - Integrate ChromaDB for high-speed local similarity search.
-  **Step 4: Search Interface**
    - Build search logic.
    - Create Streamlit UI for user interaction.

## Current Status
Currently on **Step 3**, editing/formatting/finalizing/refactoring the vector indexing file
