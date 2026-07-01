# Local Multimodal Image Search Engine

A local semantic image search engine built with CLIP and ChromaDB. 
Instead of searching by filename or tags, it understands the *content* of your images — 
type a natural language query like `"a dog on a beach"` and it retrieves the most visually 
similar images from your local folder. Runs entirely on your machine with no cloud services or API keys required.

## Project Structure
```
my_project/
├── .venv/                # Virtual environment
├── data/                 # Your image collection (outside src)
├── index/                # Local ChromaDB database
├── src/                  # Core logic (importable packages)
│   ├── __init__.py
│   ├── embedder.py
│   └── indexer.py
├── app.py                # Streamlit UI entry point
├── main.py               # CLI entry point (indexing/searching)
├── requirements.txt      # Project dependencies
└── README.md
```


## Setup & Installation

### Prerequisites
- Python 3.12 (not 3.13 — see Troubleshooting)
- NVIDIA GPU with CUDA support (optional but recommended)
- Git

---

### 1. Clone the Repository
```bash
git clone https://github.com/christophertna/multimodal-image-search.git
cd multimodal-image-search
```

---

### 2. Create a Virtual Environment with Python 3.12
```bash
py -3.12 -m venv venv
```

Activate it:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt confirming it's active.

---

### 3. Install PyTorch with CUDA Support
If you have an NVIDIA GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you are on CPU only:
```bash
pip install torch torchvision
```

Verify CUDA is detected (NVIDIA GPU users only):
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

### 4. Install Remaining Dependencies
```bash
pip install transformers pillow chromadb streamlit
```

---

### 5. Set Up Project Folders
Create the image folder where your images will live:
```bash
# Windows
mkdir data\images

# Mac/Linux
mkdir -p data/images
```

Drop any `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` images into `data/images/`.
Aim for at least 20-50 images for reasonable search quality.

---

### 6. Run the App
```bash
streamlit run app.py
```

Browser should open automatically at `http://localhost:8501`.

---

### 7. Index Your Images
1. In the browser, go to the **Index Images** tab
2. Confirm the image folder path in the sidebar matches where your images are (`./data/images` by default)
3. Click **Start Indexing**
4. Wait for the progress bar to complete — the first run also downloads the CLIP model (~350MB, 1 time only)
5. The sidebar will update showing the number of vectors stored

---

### 8. Search
1. Switch to the **Search** tab
2. Type a natural language query such as `"a photo of a sunset"` 
3. Click **Search**
4. Results appear as an image grid ranked by similarity score

---


### CLI Usage (without Streamlit)
You can also run the pipeline directly from the terminal:
```bash
# Index all images in a folder
python main.py --mode index --data_dir ./data/images

# Search with a text query
python main.py --mode search --query "a photo of a cat"

# Search and return more results
python main.py --mode search --query "a photo of a cat" --top_k 10
```


---

## In Action

<img width="1919" height="909" alt="Indexing" src="https://github.com/user-attachments/assets/9cbacb8f-7b35-4007-afa7-7827c2d2604c" />
<br>
<img width="1919" height="905" alt="Search" src="https://github.com/user-attachments/assets/efea4d14-99c6-40a3-9ceb-47df4ce24a1b" />
<br>
<img width="1918" height="907" alt="Results" src="https://github.com/user-attachments/assets/f23970f6-a7af-4187-8ac1-3862cf79caba" />
<br>
<img width="1919" height="909" alt="Results 2" src="https://github.com/user-attachments/assets/603cbdd6-bbb5-4720-81d8-4a72e242c68f" />

---


## Main Project Roadmap
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



## Troubleshooting (All issues encountered)

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
