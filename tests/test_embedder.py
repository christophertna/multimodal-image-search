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