"""
embedder.py - CLIP model wrapper for generating image and text embeddings.

Architecture note:
    CLIP (Contrastive Language-Image Pretraining) is a dual-encoder model.

    It maps both images and text into a *shared* vector space, meaning a text
    query like "a dog on a beach" and a matching photo will produce vectors
    that are geometrically close. This is what makes semantic search across
    modalities possible.

    This module is intentionally stateless beyond the loaded model. 

    It only owns ONE responsibility: raw input → normalized embedding vector.

    The indexer.py module will own persistence and retrieval.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    """
    # HINT: torch.cuda.is_available() returns True if your NVIDIA drivers
    # and CUDA toolkit are correctly installed. If this returns False on your
    # machine, you may need to reinstall PyTorch with CUDA support:
    #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Mac
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """
    Wraps the CLIP model and processor, exposing two public methods:
        - embed_text(text: str)  -> torch.Tensor
        - embed_image(image_path: str) -> torch.Tensor

    Both return L2-normalized vectors in the same shared embedding space,
    ready to be compared via cosine similarity (dot product of unit vectors).
    """
    # An L2 normalized vector (aka unit vector) is a vector whose elements 
    # have been scaled so that its overall magnitude or "length" is exactly 1. 

    # This scaling is based on the L2 norm (Euclidean distance), which measures the straight-line distance from the vector's origin


    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.device = get_device()
        print(f"[CLIPEmbedder] Using device: {self.device}")

        # HINT: CLIPProcessor bundles two sub-processors:
        #   - A tokenizer for text (converts words → token IDs)
        #   - An image processor (resizes, normalizes pixel values)
        # You don't need to configure them separately.
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)

        # HINT: .to(self.device) moves the model's weights from CPU RAM onto
        # your GPU's VRAM. All input tensors must also be on the same device
        # before you run a forward pass — that's handled below.
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)

        # Freeze weights: we are using CLIP for inference only, not fine-tuning.
        # This disables gradient tracking, saving memory and speeding up inference.
        self.model.eval()

    # -----------------------------------------------------------------------
    # Text embedding
    # -----------------------------------------------------------------------

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Convert a text string into a normalized embedding vector.

        Args:
            text: A natural-language query, e.g. "a red car on a highway".

        Returns:
            A 1D float32 tensor of shape (512,) on CPU, L2-normalized.
        """
        # HINT: The processor's job is to tokenize the raw string into
        # token IDs and an attention mask. `return_tensors="pt"` means
        # "return PyTorch tensors" (as opposed to numpy or TF).
        # `padding=True` and `truncation=True` handle variable-length inputs.
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # HINT: Move every tensor in `inputs` to the same device as the model.
        # {k: v.to(...)} is a dict comprehension — a concise way to transform
        # all values in a dictionary at once.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # HINT: torch.no_grad() is a context manager that disables autograd
        # (the automatic differentiation engine). During inference we never
        # call .backward(), so this saves memory and speeds up the forward pass.
        with torch.no_grad():
            # HINT: get_text_features() runs only the text encoder branch of
            # CLIP, returning a tensor of shape (batch_size, 512).
            # YOUR TASK: Call the right model method here and store the result.

            # ANSWER: self.model.get_text_features(**inputs)
            
            #   - `self.model` is the CLIPModel loaded in __init__
            #   - `.get_text_features()` runs only the text encoder (ignores the image encoder)
            #   - `**inputs` unpacks the dict from the processor: {"input_ids": ..., "attention_mask": ...}
            #     into keyword arguments, which is what the model method expects
            text_features = self.model.get_text_features(**inputs)

        # HINT: We take [0] to go from shape (1, 512) → (512,) since we
        # processed a single string. Then we L2-normalize so that cosine
        # similarity later reduces to a simple dot product — fast and numerically
        # stable. .cpu() moves the tensor back to RAM so it's device-agnostic
        # for the indexer.
        return self._normalize(text_features[0].cpu())

    # -----------------------------------------------------------------------
    # Image embedding
    # -----------------------------------------------------------------------

    def embed_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk and convert it into a normalized embedding vector.

        Args:
            image_path: Absolute or relative path to an image file.

        Returns:
            A 1D float32 tensor of shape (512,) on CPU, L2-normalized.
        """
        # HINT: PIL (Pillow) is the standard Python image library.
        # .convert("RGB") ensures consistent 3-channel format regardless
        # of whether the source is PNG (RGBA), grayscale, etc.
        image = Image.open(image_path).convert("RGB")

        # HINT: The processor here acts as the image branch: it resizes to
        # 224x224, normalizes pixel values to the range CLIP was trained on,
        # and returns a tensor of shape (1, 3, 224, 224).
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # YOUR TASK: Mirror what you did for text, but use the image
            # encoder branch. Look up `get_image_features` in the CLIP docs.

            # ANSWER: self.model.get_image_features(**inputs)

            #   - Same pattern as text, but `.get_image_features()` runs the vision encoder
            #   - `**inputs` unpacks {"pixel_values": ...} — the preprocessed image tensor
            #     of shape (1, 3, 224, 224) that the processor prepared above
            #   - Output shape is also (batch_size, 512), same embedding space as text
            image_features = self.model.get_image_features(**inputs)

        return self._normalize(image_features[0].cpu())

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize a 1D tensor so its magnitude (norm) equals 1.

        Why this matters:
            Cosine similarity = dot(a, b) / (||a|| * ||b||)
            If both vectors are already unit length, ||a|| = ||b|| = 1,
            so cosine similarity simplifies to just dot(a, b).
            This makes retrieval at query time a single matrix multiply.
        """
        # HINT: torch.nn.functional.normalize works along a given dimension.
        # For a 1D vector, dim=0 is the only dimension. For batched tensors
        # (2D), you would use dim=1 to normalize each row independently.
        # YOUR TASK: Import torch.nn.functional and call normalize here.

        # ANSWER: torch.nn.functional.normalize(tensor, dim=0)

        #   - `torch.nn.functional` (often aliased as `F`) contains stateless
        #     math ops like normalize, relu, softmax — no learnable parameters
        #   - `normalize(tensor, dim=0)` divides every element by the tensor's L2 norm:
        #     result = tensor / sqrt(sum(tensor[i]^2))  →  ||result|| = 1.0
        #   - No separate import needed since `torch` is already imported above;
        #     torch.nn.functional is a submodule accessible directly from it
        return torch.nn.functional.normalize(tensor, dim=0)