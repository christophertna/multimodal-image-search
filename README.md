# Local Multimodal Image Search Engine

A local semantic image search engine built with CLIP and ChromaDB. 
Instead of searching by filename or tags, it understands the *content* of your images — 
type a natural language query like `"a dog on a beach"` and it retrieves the most visually 
similar images from your local folder. Runs entirely on your machine with no cloud services or API keys required.

## Project Structure
```
my_project/
├── .venv/                # Virtual environment
├── data/                 # Image collection (outside src)
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

---

## Troubleshooting (All issues encountered):

---

### `ModuleNotFoundError: No module named 'torchvision'`

**Issue:**
After setting up the venv and installing the required dependencies, the app 
crashed immediately on startup before any code ran. The terminal showed a long chain of 
import errors originating from deep inside the `transformers` library, eventually pointing 
to a missing `torchvision` module. This was confusing because `torchvision` was never 
explicitly mentioned as a dependency, but `transformers` internally relies on it.

**Fix:**
Install `torchvision` explicitly alongside `torch`, making sure to match the CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Lesson learned:**
Large ML libraries like `transformers` have many optional sub-dependencies that aren't 
always listed upfront. 

---

### `ValueError: torch.load` vulnerability / requires torch >= 2.6

**Issue:**
After fixing the `torchvision` issue, a new error appeared when the CLIP model tried to 
load its weights from disk. The error referenced a known security vulnerability (`CVE-2025-32434`) 
in `torch.load` and stated that PyTorch 2.6 or higher was required. This happened because 
the `--index-url https://download.pytorch.org/whl/cu121` index was used initially, which 
only distributes PyTorch up to version 2.5 which is not new enough to satisfy the `transformers` 
library's security requirement.

**Fix:**
Uninstall the old version and reinstall from the `cu128` index which ships PyTorch 2.6+:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Lesson learned:**
PyTorch maintains separate package indexes per CUDA version (`cu121`, `cu124`, `cu128` etc.). 
Not all indexes carry the latest PyTorch release so always use the most recent CUDA index. 

---

### `torch.cuda.is_available()` returns `False` (Python 3.13 incompatibility)

**Issue:**
Even after reinstalling PyTorch with the CUDA index, `torch.cuda.is_available()` kept 
returning `False` and the app was loading the CLIP model on CPU instead of the GPU. 
Running the install command again returned `ERROR: Could not find a version that satisfies 
the requirement torch (from versions: none)` meaning PyTorch simply had no available build to download. 
The root cause was Python 3.13, which the entire PyTorch ML ecosystem does NOT support (as of mid 2026). 
PyTorch CUDA builds are only available for Python 3.10 through 3.12, so had to remake the venv.

**Fix:**
Recreate the virtual environment entirely using Python 3.12:
1. Download Python 3.12 from https://www.python.org/downloads/release/python-3129/
2. Click **Disable path length limit** before closing the installer (prevents obscure pip errors with long nested paths)
3. Delete the old venv and recreate it:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow chromadb streamlit
```

**Lesson learned:**
The ML ecosystem (PyTorch, TensorFlow, and most libraries built on them) seems to always 
lag 1-2 versions behind the latest Python release. For any ML project, Python 3.10-3.12 
is the safe zone (for now). Always check PyTorch's official compatibility before starting a 
new project: https://pytorch.org/get-started/locally/

---

### `ValueError: Expected a 1D embedding tensor, got shape torch.Size([1, 50, 768])`

**Issue:**
This was the MOST persistent bug in the project. After clicking Start Indexing, all 
images would embed successfully (the progress bar completed), but then the app crashed 
when trying to store them in ChromaDB. The error said it received a tensor of shape 
`(1, 50, 768)` when it expected `(512,)`. The shape `(1, 50, 768)` being 50 tokens & 768 
hidden dimensions, is the raw internal hidden state of the TRANSFORMER, not the final 
embedding vector. After ruling out issues in `app.py`, `indexer.py`, caching, and the 
virtual environment, a direct terminal test confirmed the bug was inside `embed_image()` 
itself. The cause was a breaking change in newer versions of `transformers`: 
`get_image_features()` and `get_text_features()` no longer return a plain tensor, instead they 
now return a `BaseModelOutputWithPooling` object that wraps the tensor inside named attributes.

**Fix:**
Extract the tensor from the object after each model call in `embedder.py`:
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

**Lesson learned:**
ML library APIs change between versions and don't always raise obvious errors, so sometimes 
they silently return a different type instead of crashing immediately. When debugging shape 
errors, bypass the full application and test the suspect method directly in the terminal 
to isolate the problem:
```bash
python -c "
from src.embedder import CLIPEmbedder
e = CLIPEmbedder()
t = e.embed_image('data/images/your_image.jpg')
print('shape:', t.shape)
"
```

---

### Search precision is low with single-word queries

**Issue:**
Searching with single words like `"cat"` or `"food"` produced noticeably worse results 
than longer descriptive phrases. However this is expected behavior and not a bug.

**Why it happens:**
CLIP was trained on hundreds of millions of image-text pairs where the text side was 
always a natural language DESCRIPTION like `"a photo of a cat sitting on a couch"`, 
never a single keyword. So a one-word query produces a vague, diffuse vector in the 
embedding space that doesnt point precisely toward any particular type of image. A full 
sentence produces a sharp, specific vector that lands much closer to genuinely matching images.

**Fix:**
Use natural language queries, ideally prefixed with `"a photo of"`, which is the exact phrasing 
used in CLIP's training data:
```
# Less precise
"cat"

# More precise
"a photo of a cat"
"a photo of a sunset over mountains"
"a photo of a person smiling"
```
