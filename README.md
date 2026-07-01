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
    - Built search logic
    - Created Streamlit UI for user interaction


## Project Structure
```
my_project/
├── .venv/                # Virtual environment
├── data/                 # Your image collection (outside src)
├── index/                # local ChromaDB database
├── src/                  # Core logic (importable packages)
│   ├── __init__.py
│   ├── embedder.py
│   └── indexer.py
├── app.py                # Streamlit UI entry point
├── main.py               # CLI entry point (indexing/searching)
├── requirements.txt      # Project dependencies
└── README.md
```


## Troubleshooting (All errors encountered)

### `ModuleNotFoundError: No module named 'torchvision'`
Install torchvision explicitly, matching the CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

### `ValueError: torch.load` vulnerability / requires torch >= 2.6
The `cu121` index only ships PyTorch 2.5. Switch to `cu128` which includes 2.6+:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

### `torch.cuda.is_available()` returns `False`
PyTorch CUDA builds do NOT support Python 3.13. Recreate your virtual environment using Python 3.12:
1. Download Python 3.12 from https://www.python.org/downloads/release/python-3129/
2. During install, click **Customize Installation** → on Advanced Options check **Install for all users**
3. Click **Disable path length limit** before closing the installer
4. Recreate the venv:
```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow chromadb streamlit
```

---

### `ValueError: Expected a 1D embedding tensor, got shape torch.Size([1, 50, 768])`
Newer versions of `transformers` changed `get_image_features()` and `get_text_features()` 
to return a `BaseModelOutputWithPooling` object instead of a raw tensor. 
Extract the tensor explicitly after each model call in `embedder.py`:

```python
# In embed_image()
image_features = self.model.get_image_features(**inputs)
if hasattr(image_features, 'image_embeds'):
    image_features = image_features.image_embeds
elif hasattr(image_features, 'pooler_output'):
    image_features = image_features.pooler_output

# In embed_text()
text_features = self.model.get_text_features(**inputs)
if hasattr(text_features, 'text_embeds'):
    text_features = text_features.text_embeds
elif hasattr(text_features, 'pooler_output'):
    text_features = text_features.pooler_output
```

---

### Search precision is low with single-word queries
CLIP was trained on descriptive sentences, not keywords. Use natural language 
queries for better results, ideally prefixed with `"a photo of"`:
```
# Less precise
"cat"

# More precise  
"a photo of a cat"
"a photo of a sunset over mountains"
"a photo of a person smiling"
```

## Current Status
Completed app.py, currently testing final overall application
