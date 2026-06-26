# Multimodal Image Search Engine

A local-first, high-performance semantic search engine for images using CLIP and vector embeddings.

## Project Roadmap
- [x] **Step 1: Environment Setup** - Virtual environment configured.
    - Git repository initialized.
    - Project structure created (`/src`, `/data`, `/index`).
- [x] **Step 2: Model Integration**
    - `src/embedder.py` implemented using `CLIPModel` and `CLIPProcessor`.
    - Device-agnostic code (CUDA/MPS/CPU) and L2-normalization established.
- [ ] **Step 3: Vector Indexing (In Progress)**
    - Implement `src/indexer.py` to batch-process images.
    - Integrate FAISS for high-speed local similarity search.
- [ ] **Step 4: Search Interface**
    - Build search logic.
    - Create Streamlit UI for user interaction.

## Current Status
We have successfully wrapped the CLIP model and established a clean, modular architecture. The model is currently loading successfully and generating normalized 512-dimensional vectors. We are going to move to **Step 3** once done, where we will create the indexer to catalog the image data.