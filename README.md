# Local Multimodal Image Search Engine

A local semantic image search engine built with *OpenAI*'s *CLIP* model and *ChromaDB*. 
Instead of searching by filename or tags, it understands the content of your images, just 
type a natural language query like `"a dog on a beach"` and it retrieves the most visually 
similar images from your local folder. Runs entirely on your machine with no cloud services or API keys required.

## Project Structure
```
my_project/
├── .venv/                  # Virtual environment
├── data/                   # Image collection (outside src)
├── index/                  # Local ChromaDB database
├── src/                    # Core logic (importable packages)
│   ├── __init__.py
│   ├── embedder.py
│   └── indexer.py
│
├── app.py                  # Streamlit UI entry point
├── main.py                 # CLI entry point (indexing/searching)
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker container image definition
├── docker-compose.yml      # Docker container build/run
├── .github/workflows
│   └── docker-publish.yml  # Docker container configuration
│
├── .streamlit              
│   └── config.toml         # Streamlit UI data
│
├── tests
│   ├── __init__.py
│   ├── test_embedder.py    # tests for embedding
│   └── test_indexer.py     # tests for indexing
│
├── .flake8                 # flake8 configuration
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

Should see `(venv)` at the start of your terminal prompt confirming it's active.

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
4. Wait for the progress bar to complete: 1st run also downloads the CLIP model (~350MB, 1 time only)
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


<br>

<br>

<br>

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
Large ML libraries like `transformers` have many optional sub-dependencies that arent 
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
PyTorch maintains separate package indexes per CUDA version (`cu121`, `cu124`, `cu128` etc.)
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
when trying to store them in *ChromaDB*. The error said it received a tensor of shape 
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
ML library APIs change between versions and dont always raise obvious errors, so sometimes 
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
Searching with single words like `"cat"` produced noticeably worse results 
than longer descriptive phrases like` "a picture of a cat"` . However, this is expected behavior and not a bug.

**Why it happens:**
*OpenAI's CLIP* model was trained on hundreds millions of image/text pairs where the text side was 
always a natural language DESCRIPTION like `"a photo of a cat sitting on a couch"`, 
NEVER a single keyword. So a one-word query produces a vague vector in the 
embedding space that doesnt point precisely toward any particular type of image. A full 
sentence produces a sharp, specific vector that lands much closer to genuinely matching images.

**Fix:**
Use natural language queries, ideally prefixed with `"a photo of"`, which is the exact phrasing 
used in CLIP's training data:
```
Less precise:
"cat"

More precise:
"a photo of a cat"
"a photo of a sunset over mountains"
"a photo of a person smiling"
```

---

## Docker Setup

Easiest way to run this project (no Python/venv/CUDA configuration required):
Docker handles the entire environment inside a container, but uses CPU bt default

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Images placed in `./data/images/` before starting

---

### 1. Make sure Docker Desktop is running
Open Docker Desktop from your Start menu or taskbar and wait for the green
**"Engine running"** status in the bottom left corner.

Verify it's working:
```bash
docker --version
docker run hello-world
```

---

### 2. Place your images in the images folder
```
multimodal-image-search/
└── data/
    └── images/      ← drop your image files here
```

---

### 3. Build the container image
Run this from the project root (same folder as `Dockerfile`):
```bash
docker compose build
```

This will:
- Pull the Python 3.12 base image
- Install all dependencies via pip
- Copy your project code into the image

This takes a few minutes on the first run. Subsequent builds are much faster
thanks to Docker's layer caching. As long as `requirements.txt` hasnt changed,
the pip install step is skipped entirely.

---

### 4. Start the app
```bash
docker compose up
```

Then open your browser at:
```
http://localhost:8501
```

---

### 5. Index your images
1. Go to the **Index Images** tab in the browser
2. Click **Start Indexing**
3. Wait for the process to complete

> **Note:** The first run will download the CLIP model (~350MB) inside the container. <br>
> (This only happens once so its cached in the container image afterward)

---

### 6. Search
Switch to the **Search** tab, type a natural language query, and click **Search**:
```
"a photo of a sunset"
"a photo of a dog on the beach"
"a photo of a person smiling"
```

---

### Useful Docker commands

```bash
# Start the container in the background (detached mode)
docker compose up -d

# Stop the container
docker compose down

# View live logs
docker compose logs -f

# Rebuild after code changes
docker compose build
docker compose up

# Open a shell inside the running container (for debugging)
docker exec -it image-search-app bash
```

---

### How persistence works
Two folders on your local machine are mounted into the container as volumes:

| Local folder | Container path | Purpose |
|---|---|---|
| `./data/images` | `/app/data/images` | Your images: add new ones locally and they appear instantly |
| `./index` | `/app/index` | *ChromaDB* database: survives container restarts, no re-indexing needed |

This means stopping and restarting the container with `docker compose down` and
`docker compose up` keeps your index intact.

---

### CPU vs GPU
The Docker setup runs on CPU only, which is sufficient for small to medium image
collections. Embedding speed is slower than the native GPU setup but search quality
and results are identical: the CLIP model weights are the same regardless of device.

For reference:
| Setup | Device | Indexing speed (per image) |
|---|---|---|
| Local venv (NVIDIA GPU) | CUDA | ~0.1s |
| Docker | CPU | ~0.5-1s |

---

### Docker Hub

The latest image is automatically published to Docker Hub on every successful pipeline run.

**Pull and run without cloning the repo:**
```bash
docker pull christophertan203/multimodal-image-search:latest
```

Then create a `docker-compose.yml` locally:
```yaml
services:
  app:
    image: christophertan203/multimodal-image-search:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data/images:/app/data/images
      - ./index:/app/index
    environment:
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped
```

And run it:
```bash
docker compose up
```

This lets anyone run the app without cloning the repository, installing Python,
or setting up a venv since Docker handles everything.

---

## CI/CD Pipeline

This project uses GitHub Actions for automated testing and Docker Hub for image distribution.

---

### GitHub Actions Workflow

Every push to `main` that touches application code automatically:

1. **Runs tests** — executes the full pytest suite against the embedder and indexer
2. **Builds the Docker image** — verifies the container builds successfully
3. **Pushes to Docker Hub** — publishes the latest image if all previous steps pass

The pipeline is configured with path filtering, meaning it only triggers when relevant
files change. Pushing changes to the README, documentation, or empty folders will not
trigger a build.

**Files that trigger the pipeline:**
```
src/**
app.py
main.py
Dockerfile
docker-compose.yml
requirements.txt
tests/**
.github/workflows/**
```

You can view all workflow runs under the **Actions** tab on the GitHub repository.

---

### Running Tests Locally

Make sure *pytest* is installed in your virtual environment:
```bash
pip install pytest
```

Run the full test suite:
```bash
pytest tests/ -v
```

The suite includes tests covering:
- Device detection (CUDA / MPS / CPU fallback)
- Text & image embedding shape, normalization, and dtype
- Cross-modal compatibility (text/image vectors sharing the same space)
- *ChromaDB* ID generation and determinism
- Embedding validation and dimension mismatch handling
- Upsert idempotency (re-indexing same image never creates duplicates)
- Batch insertion and length mismatch handling
- Similarity search correctness and self-similarity
- Collection reset behavior

All tests use temporary directories and generated data. The real `./index` folder
and `./data/images/` are never touched during testing.

---

### Setting Up CI/CD Secrets

For the GitHub Actions pipeline to push to Docker Hub, two repository secrets
must be configured:

1. Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:

| Secret name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | A Docker Hub access token (not your password) |

To generate a Docker Hub access token:
1. Log in to [hub.docker.com](https://hub.docker.com)
2. Go to **Account Settings** → **Personal Access Tokens**
3. Click **Generate new token**
4. Copy the token and paste it as the `DOCKERHUB_TOKEN` secret on GitHub

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
    - Integrated *ChromaDB* for high-speed local similarity search
-  **Step 4: Search Interface**
    - Built search logic
    - Created Streamlit UI for user interaction

---
