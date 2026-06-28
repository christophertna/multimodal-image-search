"""
embedder.py - CLIP model wrapper for generating image and text embeddings

Architecture note:
    CLIP (Contrastive Language-Image Pretraining) is a dual-encoder model:

    It maps both images and text into a SHARED vector space, meaning a text
    query and a matching photo will produce vectors that are geometrically close. 

    This is what makes semantic search across modalities possible (what about video & audio too?)

    It only has 1 responsibility: raw input (text/image) → normalized embedding vector

    The indexer.py will be the one to own persistence and retrieval.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Device selection
def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    """
    # torch.cuda.is_available() returns True if the NVIDIA drivers and CUDA toolkit are correctly installed on the setup. 
    # If it returns False, may need to reinstall PyTorch with CUDA support:

    #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Mac
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Embedder class
class CLIPEmbedder:
    """
    Wraps the CLIP model and processor, exposing two public methods:
        - embed_text(text: str)  -> torch.Tensor
        - embed_image(image_path: str) -> torch.Tensor

    Both return L2-normalized vectors in the same shared embedding space,
    ready to be compared via cosine similarity (dot product of unit vectors).
    """
    # An L2 normalized vector (aka unit vector) is a vector whose elements have been scaled so that its overall magnitude or "length" is exactly 1. 
    # Scaling is based on the L2 norm (Euclidean distance), which measures the SLD from the vector's origin.

    # Need "length" of vectors to be all squished to 1 to simplify/optimize COSINE SIMILARITY (during the semantic search/comparing)

    # Recall cosine similarity formula:  (A ⋅ B) / ||A|| * ||B||, so dot product of both vectors divided by product of the mangitude of both.
    # With the magnitude both being 1, cosine similarity essentially just becomes the original numertor operation, the dot product of both vectors.

    MODEL_ID = "openai/clip-vit-base-patch32" # free openai pretrained CLIP model

    def __init__(self):
        self.device = get_device() # run torch.device to use GPU instead (if any)
        print(f"[CLIPEmbedder] Using device: {self.device}") # show what device will be used

        # CLIPProcessor bundles two sub-processors:
        #   - Tokenizer for text (words → token IDs)
        #   - Image processor (resizes & normalizes pixel values)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID) # configured together

        # All input tensors must be on the same device before you run a forward pass, so:
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device) # move model weights from CPU RAM to GPU VRAM (if any)

        self.model.eval() # using model for predicting/inference, not training, so we "freeze"/"lock" the weights (also optimizes memory)

        # in short, CLIPProcessor handles the input, like the text query and the image conversion and 
        # the CLIPModel handles the actual calculation to convert the inputs into tensors


    # Text embedding
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Converts a text string into a normalized embedding vector:

        Takes in a natural-language input and returns a 1D float32 tensor of shape (512,) on CPU, L2-normalized
        """
        # CLIPProcessor's job is to tokenize the raw string into token IDs and 
        # an attention mask ( extra 'padding' for text input such as 1s(important token) and 0s (ignore token) )

        # 'inputs' is a dictionnary of tensors
        inputs = self.processor(
            text=[text],         # input text data as a single list
            return_tensors="pt", # return PyTorch tensors (as opposed to numpy or TF)
            padding=True,        # ensure all input text sequences have same length, here its 77 tokens for CLIP (attention mask for filling)
            truncation=True,     # cutoff exceeding text if tokens exceeds model limit
        ) 

        # processor packages these into a dictionary of tensors (as said above), but because the model expects a "batch",
        # it adds a dimension: (1, 77), where the 1 represents the batch size of 1
        # in other words, we have 1 row of 77 columns

        # IMPORTANT: we need to add the extra dimension to convert the 1D tensor (vector) into a 2D tensor (matrix) for matrix multiplication

        # Move every tensor in `inputs` to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # {k: v.to(...)} is a dict comprehension, a concise way to transform all values in a dictionary at once

        # torch.no_grad() is a context manager that disables autograd (automatic differentiation engine)
        # During inference we never call .backward(), since we are not training it, we dont need backpropagation (which needs to track every math operation on the tensors)
        with torch.no_grad(): # no need to store all computational operations during the deep learning process, so this saves memory

            # get_text_features() runs only the text encoder branch of CLIP, returning a tensor of shape (batch_size, 512), which is a 2D tensor
            text_features = self.model.get_text_features(**inputs) # **: double asterisk for tensor dicionnary unpacking (input ids and masks)

        # Then take [0] to go from shape (1, 512) --> (512,) (removing the batch container) --> 2D tensor into a 1D tensor
        # SYNTAX: (512,) the extra comma is to tell Python that this is a TUPLE containing 1 item, the number 512, which is a 1D tensor 

        # Then we L2-normalize so that cosine similarity simplifies it to a single dot product (as mentionned earlier)

        # finally .cpu() moves the tensor back to RAM so it's device-agnostic for the indexer
        return self._normalize(text_features[0].cpu())


    # Image embedding
    def embed_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk & convert it into a normalized embedding vector:

        Takes an absolute or relative path to an image file (image_path) and returns
        a 1D float32 tensor of shape (512,) on CPU, L2-normalized
        """

        # PIL (Pillow) is the standard Python image library

        # First standardize the images: ensure all images, regardless of format, is converted into a consistent 3-channel RGB format (.convert("RGB"))
        image = Image.open(image_path).convert("RGB") # every pixel will in the image will have 3 values, ex: (255, 0, 0) is pure red
        # basically cleaning/sanitizing the images beforehand

        # for CLIP, it expects 3 inputs: (3, 224, 224) where 3 is the RGB channels and both 224 is the height x width in pixels (CLIP resizes images to this)

        # The processor here acts as the image branch: it resizes to 224x224, normalizes pixel values to the range CLIP was trained on,
        # and returns a tensor of shape (1, 3, 224, 224)

        # recall the leading 1 of the tensor is the BATCH size, basically telling the model to process 1 image/"batch" at a time

        inputs = self.processor(images=image, return_tensors="pt") # like text embedding, return PyTorch tensors
        inputs = {k: v.to(self.device) for k, v in inputs.items()} # like text embedding, creating tensor dictionnary & move every tensor into same device as model

        # same as text embedding, no need to keep track of all computational operations, since we are not training the model (no backpropagation), so saving memory
        with torch.no_grad(): 
 
            # .get_image_features()" runs the vision encoder only
            image_features = self.model.get_image_features(**inputs) # recall ** unpacks the image tensor
            #  Output shape is also (batch_size, 512), a 2D tensor, same embedding space as text

        # same as text embedding, remove batch container, transfer to cpu and then L2-normalize 
        return self._normalize(image_features[0].cpu())


    # Internal helpers
    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize a 1D tensor so its magnitude/"length" (norm) equals to exactly 1
        while keeping it pointing to the same exact direction
        """
        # PyTorch Normalize works along a given dimension:
        # For a 1D vector, dim=0 is the only dimension, aka look "down" at the columns
        # For batched tensors (2D), you would use dim=1 to normalize each row independently, aka look "across" the rows 

        # "torch.nn.functional" (submodule known as 'F') contains math operations like Normalize, ReLU and Softmax, for deep learning training 

        # "normalize(tensor, dim=0)" divides every element by the tensor's L2 norm:
        #  result = tensor / sqrt(sum(tensor[i]^2))  -->  ||result|| = 1.0
        return torch.nn.functional.normalize(tensor, dim=0)
    
    # in other words:
    # 1- during embedding (before normalizing tensor), you have a 2D tensor like this: (1, 512) 
    # 2- text_features[0] removes the batch, leaving behind (512,) a 1D tensor (a tensor holding the vector basically) 
    # 3- move it to the RAM with .cpu(), so now accessible to Python or CPU-based libraries
    # 4- normalize(tensor, dim=0) is called, and since we input a 1D tensor (512,) only the dim=0 exists for it (only valid axis for it),
    #    since it treats the entire 1D array as a unit and scales its magnitude/norm to exactly 1.0 

