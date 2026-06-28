# Multimodal Image Search Engine

A local-first, high-performance semantic search engine for images using CLIP and vector embeddings.

## Project Roadmap
-  **Step 1: Environment Setup** 
    - Git repository initialized
    - Project structure created (`/src`, `/data`, `/index`)
    - Virtual environment configured
-  **Step 2: Model Integration**
    - `src/embedder.py` implemented using `CLIPModel` and `CLIPProcessor`
    - Device-agnostic code (CUDA/MPS/CPU) and L2-normalization established
-  **Step 3: Vector Indexing**
    - Implemented `src/indexer.py` to batch-process images
    - Integrated ChromaDB for high-speed local similarity search
-  **Step 4: Search Interface**
    - Build search logic
    - Create Streamlit UI for user interaction


## Project Structure
```
my_project/
├── .venv/                # Virtual environment
├── data/                 # Your image collection (outside src)
├── src/                  # Core logic (importable packages)
│   ├── __init__.py
│   ├── embedder.py
│   └── indexer.py
├── app.py                # Streamlit UI entry point
├── main.py               # CLI entry point (indexing/searching)
├── requirements.txt      # Project dependencies
└── README.md
```


## Current Status
Finished **Step 3**, now onto main.py (combine embedder + indexer) and later on app.py (Step 4)
