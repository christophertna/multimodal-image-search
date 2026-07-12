"""
tests/test_embedder.py (smoke tests for Embedder class)

What are smoke tests?
    Smoke tests are very simple tests, they verify that core
    components load & run without crashing. They dont test edge cases
    or business logic, just checks that fundamental wiring works

Why not test more deeply?
    CLIP is a pretrained model from Hugging Face. We didnt write the model
    itself, so we dont need to test that it produces correct embeddings,
    which is OpenAI's responsibility. 
    
    What we need to test are:
        1. Our wrapper loads the model correctly
        2. Our wrapper produces the right output shape and type
        3. The normalization is applied correctly

    Run tests locally with:
        pytest tests/ -v
"""

import torch
import pytest
from PIL import Image
import tempfile
import os

from src.embedder import CLIPEmbedder


# Fixtures
# 
# A fixture is a reusable setup function. Instead of loading the
# CLIPEmbedder in every single test (very slow), define it once as a fixture
# and pytest injects it into any test function that lists it as a parameter
#
# scope="module" means the fixture is created ONCE for the entire test file
# and reused across all tests, so the model loads once, not per test.
# (Critical for expensive ML models)

@pytest.fixture(scope="module")
def embedder():
    """Load CLIPEmbedder once and share it across all tests in this file."""
    return CLIPEmbedder()


@pytest.fixture(scope="module")
def sample_image_path():
    """
    Create a temporary 224x224 RGB image on disk for testing.

    Why a temporary image?
        Dont want tests to depend on specific files existing in 'data/images/'
        since that folder isnt committed to the repo. A temporary image is
        created fresh for EACH test session and deleted automatically afterward.

    "tempfile.NamedTemporaryFile" creates a file that is automatically cleaned up
    when the context manager exits, so no leftover test files on disk.
    """
    # Create a simple solid red 224x224 image, content doesnt matter for
    # smoke tests, we just need a valid image file that PIL can open:
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))

    # suffix=".jpg": ensures the file has the right extension so our
    # 'SUPPORTED_EXTENSIONS' check in app.py would accept it
    #
    # "delete=False" on Windows (tempfile cant delete open files on Windows)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name)
        path = f.name

    yield path  # provide the path to the test

    # Cleanup: delete temp file after all tests in this module finish
    # (yield + cleanup below is pytests pattern for teardown logic)
    os.unlink(path)


# Device tests
def test_device_is_valid(embedder):
    """
    Verify 'get_device()' returns a valid torch.device object

    Why test this?
        If CUDA isnt available, app should gracefully fall back to CPU
        rather than crashing (confirms fallback logic)
    """
    assert isinstance(embedder.device, torch.device)
    assert str(embedder.device) in ("cuda", "mps", "cpu")


# Text embedding tests
def test_embed_text_returns_tensor(embedder):
    """embed_text() should return a PyTorch tensor."""
    result = embedder.embed_text("a dog on a beach") # example text query
    assert isinstance(result, torch.Tensor)


def test_embed_text_shape(embedder):
    """
    embed_text() should return a 1D tensor of exactly 512 dimensions.

    Why 512?
        CLIP ViT-B/32 projects all embeddings (text/image) into a
        512-dimensional shared space. This is a fixed architectural property
        of the model, so if this fails, model loaded incorrectly
    """
    result = embedder.embed_text("a dog on a beach") # example text query
    assert result.shape == torch.Size([512]), \
        f"Expected shape (512,) but got {result.shape}"


def test_embed_text_is_normalized(embedder):
    """
    embed_text() should return an L2-normalized vector (unit vector).

    Why test normalization?
        Cosine similarity search in ChromaDB relies on vectors being
        unit length. If normalization is broken, similarity scores will be
        wrong & search results will be meaningless.

    How to verify:
        The L2 norm of a unit vector is exactly 1.0.
        torch.norm() computes the L2 norm (allow small tolerance (1e-6)
        for floating point rounding errors)
    """
    result = embedder.embed_text("a sunset over mountains") # example text query
    norm = torch.norm(result).item()
    assert abs(norm - 1.0) < 1e-6, \
        f"Expected unit norm (1.0) but got {norm}. Is _normalize() working?"


def test_embed_text_on_cpu(embedder):
    """embed_text() should always return a CPU tensor for device-agnostic indexing."""
    result = embedder.embed_text("a photo of a cat") # example text query
    assert result.device.type == "cpu", \
        "Embedding should be moved to CPU before returning (.cpu() call)"


def test_embed_text_different_inputs_differ(embedder):
    """
    Different text inputs should produce different embedding vectors.
    (Sanity check to ensure model is actually processing input text)
    """

    vec1 = embedder.embed_text("a dog on a beach") # example text query 1
    vec2 = embedder.embed_text("a red sports car") # example text query 2
    assert not torch.allclose(vec1, vec2), \
        "Different text inputs should produce different embeddings"


# Image embedding tests
def test_embed_image_returns_tensor(embedder, sample_image_path):
    """embed_image() should return a PyTorch tensor."""
    result = embedder.embed_image(sample_image_path)
    assert isinstance(result, torch.Tensor)


def test_embed_image_shape(embedder, sample_image_path):
    """embed_image() should return a 1D tensor of exactly 512 dimensions."""
    result = embedder.embed_image(sample_image_path)
    assert result.shape == torch.Size([512]), \
        f"Expected shape (512,) but got {result.shape}"


def test_embed_image_is_normalized(embedder, sample_image_path):
    """embed_image() should return an L2-normalized vector."""
    result = embedder.embed_image(sample_image_path)
    norm = torch.norm(result).item()
    assert abs(norm - 1.0) < 1e-6, \
        f"Expected unit norm (1.0) but got {norm}. Is _normalize() working?"


def test_embed_image_on_cpu(embedder, sample_image_path):
    """embed_image() should always return a CPU tensor."""
    result = embedder.embed_image(sample_image_path)
    assert result.device.type == "cpu"


def test_embed_image_invalid_path(embedder):
    """
    embed_image() should raise an error when given a non-existent file path.

    pytest.raises() is a context manager that asserts an exception IS raised.
    If the code inside does NOT raise the expected exception, the test fails.
    This tests our error handling (app should fail clearly)
    """
    with pytest.raises(Exception):
        embedder.embed_image("non_existent_image.jpg")


# Multimodal tests
def test_text_and_image_same_space(embedder, sample_image_path):
    """
    Text/image embeddings should live in the SAME vector space, with
    both returning 512-dim unit vectors that can be compared directly

    If text returns (512,) and image returns (256,), cosine similarity
    between them would be undefined & search would be broken
    """

    text_vec  = embedder.embed_text("a red image") # example text query
    image_vec = embedder.embed_image(sample_image_path) # example image path

    assert text_vec.shape == image_vec.shape, \
        "Text and image embeddings must have the same shape for cosine similarity"
    assert text_vec.dtype == image_vec.dtype, \
        "Text and image embeddings must have the same dtype"
    
# New tests that werent covered earlier:
#   - Embeddings are deterministic (same input -> same
#     output every time which confirms the model is truly in eval mode with
#     no leftover randomness like dropout)
#   - embed_image() on a non-RGB image, despite embedder.py
#     explicitly calling .convert("RGB") to handle exactly that case
#   - Text longer than model limit (than CLIP's 77-token limit), despite
#     truncation=True being explicitly set in the processor call
#   - Empty string as input
#   - Model's weights live on the device
#   - Embeddings carry any actual semantic meaning, every
#     existing test only checks shape/dtype/normalization, which would
#     all still pass even if the encoders were silently swapped or
#     returned garbage that happened to be the right shape

def test_embed_text_is_deterministic(embedder):
    """
    Calling embed_text() twice with the identical input should produce
    identical output — the model is in eval() mode (see embedder.py's
    __init__), so there should be zero randomness (no dropout, etc.)
    left active during inference.
    """
    vec1 = embedder.embed_text("a photo of a mountain")
    vec2 = embedder.embed_text("a photo of a mountain")
    assert torch.allclose(vec1, vec2), \
        "Same text input should always produce the same embedding (model should be fully deterministic in eval mode)"


def test_embed_image_is_deterministic(embedder, sample_image_path):
    """Same as above, but for embed_image()."""
    vec1 = embedder.embed_image(sample_image_path)
    vec2 = embedder.embed_image(sample_image_path)
    assert torch.allclose(vec1, vec2), \
        "Same image input should always produce the same embedding"


def test_embed_image_handles_non_rgb(embedder, tmp_path):
    """
    embed_image() should work on non-RGB images (grayscale, RGBA, etc.),
    not just plain RGB — embedder.py explicitly calls .convert("RGB") for
    exactly this reason, but nothing in the original suite actually
    exercises that line with a non-RGB source image.
    """
    grayscale_path = tmp_path / "grayscale.jpg"
    Image.new("L", (224, 224), 128).save(grayscale_path)  # "L" = 8-bit grayscale

    result = embedder.embed_image(str(grayscale_path))

    assert result.shape == torch.Size([512])
    assert abs(torch.norm(result).item() - 1.0) < 1e-6


def test_embed_text_handles_long_text(embedder):
    """
    CLIP's tokenizer has a hard 77-token limit. embedder.py sets
    truncation=True specifically to avoid crashing on longer input, but
    that line was never actually exercised by any existing test — every
    example text query so far has been a short handful of words.
    """
    long_text = "a photo of " + "a very large mountain landscape with trees and rivers and clouds " * 20
    result = embedder.embed_text(long_text)

    assert result.shape == torch.Size([512])
    assert abs(torch.norm(result).item() - 1.0) < 1e-6


def test_embed_text_handles_empty_string(embedder):
    """An empty string is a valid (if degenerate) input — should not crash,
    should still return a properly-shaped, normalized vector."""
    result = embedder.embed_text("")
    assert result.shape == torch.Size([512])
    assert abs(torch.norm(result).item() - 1.0) < 1e-6


def test_model_weights_on_correct_device(embedder):
    """
    embedder.device reports where things SHOULD be, but nothing actually
    confirms the model's weights were moved there — if .to(self.device)
    were accidentally dropped from __init__, embedder.device would still
    report correctly while the model silently stayed on its default
    device. Checking an actual parameter's device closes that gap.
    """
    model_device = next(embedder.model.parameters()).device
    assert model_device.type == embedder.device.type, \
        f"Model parameters live on {model_device} but embedder.device claims {embedder.device}"


def test_cross_modal_semantic_similarity(embedder, tmp_path):
    """
    The most important property CLIP is actually supposed to have: a
    text description should be MORE similar to an image that matches it
    than to one that doesn't. Every other test in this file only checks
    shape/dtype/normalization — all of which would still pass even if the
    text and image encoders were silently swapped, or returned
    plausible-looking noise. This is the one test that would actually
    catch that class of bug.

    NOTE: solid-color synthetic images are a somewhat artificial,
    out-of-distribution input for a model trained on natural photos —
    this should still pass reliably given how strong CLIP's color/concept
    association is, but if it ever flakes, swapping in real photos (e.g.
    an actual photo of the sky vs. an actual photo of grass) instead of
    flat color swatches would be a more realistic, even safer bet.
    """
    red_path = tmp_path / "red.jpg"
    blue_path = tmp_path / "blue.jpg"
    Image.new("RGB", (224, 224), (220, 20, 20)).save(red_path)
    Image.new("RGB", (224, 224), (20, 20, 220)).save(blue_path)

    red_image_vec = embedder.embed_image(str(red_path))
    blue_image_vec = embedder.embed_image(str(blue_path))
    red_text_vec = embedder.embed_text("a photo of the color red")
    blue_text_vec = embedder.embed_text("a photo of the color blue")

    # Both vectors are already unit-normalized, so dot product == cosine similarity
    red_text_to_red_image = torch.dot(red_text_vec, red_image_vec).item()
    red_text_to_blue_image = torch.dot(red_text_vec, blue_image_vec).item()
    blue_text_to_blue_image = torch.dot(blue_text_vec, blue_image_vec).item()
    blue_text_to_red_image = torch.dot(blue_text_vec, red_image_vec).item()

    assert red_text_to_red_image > red_text_to_blue_image, \
        "'a photo of the color red' should be more similar to a red image than a blue one"
    assert blue_text_to_blue_image > blue_text_to_red_image, \
        "'a photo of the color blue' should be more similar to a blue image than a red one"