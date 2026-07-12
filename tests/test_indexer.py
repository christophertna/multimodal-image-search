"""
tests/test_indexer.py (smoke tests for Indexer class)

These tests use a temporary ChromaDB directory so they never touch
the real './index' folder. Every test run starts with a clean slate
and cleans up after itself automatically

Test isolation concept:
    Tests should never depend on each other or share state.
    Each test should set up what it needs and clean up after itself.
    This is why we use a temp directory for ChromaDB instead of './index'
"""

import torch
import pytest
import tempfile
import shutil

from src.indexer import VectorIndexer


# Constants
EMBEDDING_DIM = 512


# Helpers
def make_random_embedding(dim: int = EMBEDDING_DIM) -> torch.Tensor:
    """
    Generate a random normalized 1D tensor to use as a fake embedding

    Why random embeddings in tests?
        Dont need real CLIP embeddings to test the indexer, just
        need valid tensors of the right shape. Random normalized vectors
        are faster to generate and dont require loading the actual model.

    'torch.randn()' generates a tensor with values from a standard normal
    distribution (mean=0, std=1). We normalize it to make it a unit vector
    """
    vec = torch.randn(dim)
    return torch.nn.functional.normalize(vec, dim=0)


# Fixtures
#
# recall: fixtures are a reusable setup function
@pytest.fixture
def temp_indexer():
    """
    Create a VectorIndexer backed by a temporary directory.

    Why a temp directory?
        If we used './index', tests would pollute your real database with
        fake test data. A temp directory is created fresh for each test
        and deleted automatically after (test isolation)

    Fixture here has no scope argument --> defaults to "function" scope,
    meaning a brand new temp indexer is created for EACH test function.
    (Ensures tests dont affect each other)
    """
    # 'tempfile.mkdtemp()' creates a temporary directory and returns its path
    temp_dir = tempfile.mkdtemp()

    indexer = VectorIndexer(
        persist_dir=temp_dir,
        collection_name="test_collection"
    )

    yield indexer  # provide the indexer to the test

    # Teardown: delete the entire temp directory after the test finishes
    # 'shutil.rmtree()' deletes a directory and all its contents recursively
    shutil.rmtree(temp_dir, ignore_errors=True)


# Initialization tests
def test_indexer_initializes(temp_indexer):
    """VectorIndexer should initialize without errors."""
    assert temp_indexer is not None


def test_indexer_starts_empty(temp_indexer):
    """A fresh indexer should have 0 vectors stored."""
    assert temp_indexer.count() == 0


# ID generation tests
def test_path_to_id_is_deterministic():
    """
    Same file path should always produce the same ID.

    Why test determinism?
        If '_path_to_id()' returned different values each call, re-indexing
        would create duplicates instead of updating existing entries
    """
    path = "./data/images/test_image.jpg"
    id1 = VectorIndexer._path_to_id(path)
    id2 = VectorIndexer._path_to_id(path)
    assert id1 == id2, "Same path must always produce the same ID"


def test_different_paths_produce_different_ids():
    """Different file paths should produce different IDs."""
    id1 = VectorIndexer._path_to_id("./data/images/cat.jpg")
    id2 = VectorIndexer._path_to_id("./data/images/dog.jpg")
    assert id1 != id2, "Different paths must produce different IDs"


# Validation tests
def test_validate_embedding_accepts_correct_shape(temp_indexer):
    """_validate_embedding() should accept a 1D tensor of dim 512."""
    vec = make_random_embedding()
    result = temp_indexer._validate_embedding(vec)
    assert isinstance(result, list)
    assert len(result) == EMBEDDING_DIM


def test_validate_embedding_rejects_2d_tensor(temp_indexer):
    """
    '_validate_embedding()' should raise ValueError for a 2D tensor.

    Catches the exact bug encountered during development when the model
    returned (1, 512) instead of (512,) before the [0] indexing fix.
    """
    bad_tensor = torch.randn(1, EMBEDDING_DIM)  # 2D: WRONG SHAPE
    with pytest.raises(ValueError, match="1D"):
        temp_indexer._validate_embedding(bad_tensor)


def test_validate_embedding_rejects_wrong_dim(temp_indexer):
    """_validate_embedding() should raise ValueError for wrong dimension"""
    bad_tensor = torch.randn(768)  # 768-dim (BERT size, not CLIP size)
    with pytest.raises(ValueError, match="dimension"):
        temp_indexer._validate_embedding(bad_tensor)


# Insertion tests
def test_add_single_embedding(temp_indexer):
    """add() should store one embedding & increment count to 1"""
    embedding = make_random_embedding()
    doc_id = temp_indexer.add(embedding, "./data/images/cat.jpg")

    assert isinstance(doc_id, str)
    assert len(doc_id) > 0
    assert temp_indexer.count() == 1


def test_add_stores_metadata(temp_indexer):
    """add() should store the image_path in metadata"""
    embedding = make_random_embedding()
    image_path = "./data/images/cat.jpg"
    temp_indexer.add(embedding, image_path, metadata={"category": "animals"})

    # Retrieve stored item directly from ChromaDB to verify
    stored = temp_indexer.collection.get(
        ids=[VectorIndexer._path_to_id(image_path)],
        include=["metadatas"]
    )
    meta = stored["metadatas"][0]
    assert meta["image_path"] == image_path
    assert meta["category"] == "animals"


def test_add_is_idempotent(temp_indexer):
    """
    Adding the same image twice should not create duplicates.

    Testing the upsert behavior, the core of the re-indexing safety.
    If fail, clicking "Start Indexing" twice would double all entries.
    """
    embedding = make_random_embedding()
    path = "./data/images/cat.jpg"

    temp_indexer.add(embedding, path)
    temp_indexer.add(embedding, path)  # same path, same ID → should upsert/update, not duplicate

    assert temp_indexer.count() == 1, \
        "Adding the same image twice should upsert, not duplicate"


def test_add_batch(temp_indexer):
    """add_batch() should store multiple embeddings in one call"""
    embeddings = [make_random_embedding() for _ in range(5)]
    paths = [f"./data/images/image_{i}.jpg" for i in range(5)]

    ids = temp_indexer.add_batch(embeddings, paths)

    assert len(ids) == 5
    assert temp_indexer.count() == 5


def test_add_batch_length_mismatch(temp_indexer):
    """add_batch() should raise ValueError if embeddings and paths differ in length"""
    embeddings = [make_random_embedding() for _ in range(3)]
    paths = ["./data/images/image_0.jpg", "./data/images/image_1.jpg"]  # only 2 images

    with pytest.raises(ValueError):
        temp_indexer.add_batch(embeddings, paths)


# Search tests
def test_search_returns_results(temp_indexer):
    """search() should return results after indexing at least one embedding"""
    embedding = make_random_embedding()
    temp_indexer.add(embedding, "./data/images/cat.jpg")

    results = temp_indexer.search(embedding, top_k=1)

    assert len(results) == 1
    assert "image_path" in results[0]
    assert "distance" in results[0]
    assert "metadata" in results[0]


def test_search_empty_collection_raises(temp_indexer):
    """search() should raise RuntimeError if/when the collection is empty"""
    query = make_random_embedding()

    with pytest.raises(RuntimeError, match="empty"):
        temp_indexer.search(query, top_k=1)


def test_search_top_k_respected(temp_indexer):
    """search() should return at most top_k results."""

    # Index 10 embeddings
    for i in range(10):
        temp_indexer.add(make_random_embedding(), f"./data/images/img_{i}.jpg")

    results = temp_indexer.search(make_random_embedding(), top_k=3)
    assert len(results) == 3


def test_search_self_similarity(temp_indexer):
    """
    Searching with the exact same vector that was indexed should return
    that vector as the top result with distance close to 0.

    This is the most important search test, it verifies that the indexer
    can actually retrieve what was stored, and that cosine distance is
    working correctly (0 = identical vectors)
    """
    embedding = make_random_embedding()
    path = "./data/images/cat.jpg"
    temp_indexer.add(embedding, path)

    # Add some random vectors so there are multiple results to choose from
    for i in range(5):
        temp_indexer.add(make_random_embedding(), f"./data/images/noise_{i}.jpg")

    results = temp_indexer.search(embedding, top_k=1)

    assert results[0]["image_path"] == path, \
        "Searching with the exact stored vector should return that vector as top result"
    assert results[0]["distance"] < 0.01, \
        f"Self-similarity distance should be near 0, got {results[0]['distance']}"


# Utility tests
def test_reset_clears_collection(temp_indexer):
    """reset() should wipe all stored vectors and return count to 0"""
    for i in range(3):
        temp_indexer.add(make_random_embedding(), f"./data/images/img_{i}.jpg")

    assert temp_indexer.count() == 3

    temp_indexer.reset()

    assert temp_indexer.count() == 0, \
        "After reset(), collection should be empty"


# New tests

@pytest.fixture
def temp_dir():
    """
    Yields a bare temp directory path (not yet a VectorIndexer) — used by
    the persistence tests below, which need to create MULTIPLE
    VectorIndexer instances pointing at the SAME directory to simulate an
    app restart. The temp_indexer fixture above can't be reused for this
    since it only ever wraps a single indexer instance.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Persistence across restarts
def test_persists_across_reconnect(temp_dir):
    """Data added via one VectorIndexer instance should still be there
    when a NEW instance is created pointing at the same persist_dir —
    simulates closing and reopening the app."""
    first = VectorIndexer(persist_dir=temp_dir, collection_name="test_collection")
    embedding = make_random_embedding()
    first.add(embedding, "./data/images/cat.jpg", metadata={"category": "animals"})
    assert first.count() == 1

    # Brand new instance, same directory, simulates an app restart
    second = VectorIndexer(persist_dir=temp_dir, collection_name="test_collection")
    assert second.count() == 1, \
        "A new VectorIndexer pointed at the same persist_dir should see previously stored data"

    results = second.search(embedding, top_k=1)
    assert results[0]["image_path"] == "./data/images/cat.jpg"
    assert results[0]["metadata"]["category"] == "animals"


def test_reconnect_with_different_collection_name_is_isolated(temp_dir):
    """Two collections in the SAME persist_dir but with different names
    should not see each other's data — collections are meant to be
    independent 'tables', per indexer.py's own architecture note."""
    indexer_a = VectorIndexer(persist_dir=temp_dir, collection_name="collection_a")
    indexer_a.add(make_random_embedding(), "./data/images/a.jpg")

    indexer_b = VectorIndexer(persist_dir=temp_dir, collection_name="collection_b")
    assert indexer_b.count() == 0, \
        "A differently-named collection in the same persist_dir should start empty"


# search()'s filter parameter 
def test_search_filter_restricts_results(temp_indexer):
    """search() with a `filter` (metadata where-clause) should only return
    documents matching that filter, even if closer matches exist outside it."""
    target = make_random_embedding()
    temp_indexer.add(target, "./data/images/cat.jpg", metadata={"category": "animals"})
    temp_indexer.add(target, "./data/images/car.jpg", metadata={"category": "vehicles"})

    results = temp_indexer.search(
        target, top_k=5, filter={"category": {"$eq": "animals"}}
    )

    assert len(results) == 1
    assert results[0]["image_path"] == "./data/images/cat.jpg"


def test_search_filter_matching_nothing_returns_empty(temp_indexer):
    """A filter that matches no documents should return an empty list,
    not raise an error."""
    temp_indexer.add(make_random_embedding(), "./data/images/cat.jpg", metadata={"category": "animals"})

    results = temp_indexer.search(
        make_random_embedding(), top_k=5, filter={"category": {"$eq": "nonexistent"}}
    )

    assert results == []


# add_batch() gaps 
def test_add_batch_is_idempotent(temp_indexer):
    """Running add_batch() twice with the same paths should upsert, not
    duplicate — only single add() idempotency was tested previously, but
    app.py's index_images() always calls add_batch(), so THAT'S the path
    that actually needs to be idempotency-safe for re-indexing to work."""
    embeddings = [make_random_embedding() for _ in range(3)]
    paths = [f"./data/images/img_{i}.jpg" for i in range(3)]

    temp_indexer.add_batch(embeddings, paths)
    temp_indexer.add_batch(embeddings, paths)  # same paths again

    assert temp_indexer.count() == 3, \
        "Re-running add_batch() on the same paths should upsert, not duplicate"


def test_add_batch_without_metadata(temp_indexer):
    """add_batch() should work with metadatas=None (the default) and still
    store image_path for each entry."""
    embeddings = [make_random_embedding() for _ in range(2)]
    paths = ["./data/images/a.jpg", "./data/images/b.jpg"]

    ids = temp_indexer.add_batch(embeddings, paths)  # no metadatas argument

    stored = temp_indexer.collection.get(ids=ids, include=["metadatas"])
    stored_paths = {m["image_path"] for m in stored["metadatas"]}
    assert stored_paths == set(paths)


def test_add_batch_empty_lists_raises(temp_indexer):
    """
    add_batch([], []) currently raises ValueError rather than gracefully
    no-op'ing (discovered while writing this test, not the assumed ideal
    behavior). app.py's index_images() never hits this path today, since
    it already guards against empty folders before ever calling
    add_batch() — but the indexer's own public API isn't safe against
    being called this way directly. Worth an early-return guard in
    indexer.py if add_batch([], []) should no-op instead; this test just
    pins down what it currently does so a future change is a deliberate
    choice, not a silent regression either way.
    """
    with pytest.raises(ValueError):
        temp_indexer.add_batch([], [])


# search() top_k edge case 
def test_search_top_k_exceeds_collection_size(temp_indexer):
    """Requesting more results than exist in the collection should return
    whatever IS available, not raise an error or hang."""
    for i in range(3):
        temp_indexer.add(make_random_embedding(), f"./data/images/img_{i}.jpg")

    results = temp_indexer.search(make_random_embedding(), top_k=10)

    assert len(results) == 3, "Should return all 3 available results, not error on top_k > count"


# random_id() 
def test_random_id_returns_string():
    """_random_id() should return a non-empty string (a UUID)."""
    result = VectorIndexer._random_id()
    assert isinstance(result, str)
    assert len(result) > 0


def test_random_id_is_unique_across_calls():
    """Repeated calls to _random_id() should not produce the same value —
    unlike _path_to_id() (deterministic by design), this one MUST vary,
    since it exists specifically for cases with no stable path to hash."""
    ids = {VectorIndexer._random_id() for _ in range(100)}
    assert len(ids) == 100, "Every call should produce a distinct ID"


# collection.get(include=["embeddings"]) round-trip:
# entire Analytics tab depends completely on this bulk-retrieval path returning the correct stored vectors.
# it never calls search(), just fetches everything at once for the cosine-matri math
def test_get_with_include_embeddings_returns_stored_vector(temp_indexer):
    """collection.get(include=["embeddings"]) should return the EXACT
    vector that was originally stored, not just count/metadata."""
    embedding = make_random_embedding()
    temp_indexer.add(embedding, "./data/images/cat.jpg")

    stored = temp_indexer.collection.get(include=["embeddings", "metadatas"])

    assert len(stored["embeddings"]) == 1
    # ChromaDB returns embeddings as float64 on retrieval, even though the
    # original tensor was float32 — cast before comparing, or
    # torch.allclose() raises a dtype mismatch error instead of comparing
    retrieved_vector = torch.tensor(stored["embeddings"][0], dtype=torch.float32)
    assert torch.allclose(retrieved_vector, embedding, atol=1e-5), \
        "Retrieved embedding should match what was originally stored"


def test_get_with_include_embeddings_returns_all_vectors(temp_indexer):
    """Same round-trip check but for multiple stored images — the
    Analytics tab always fetches ALL embeddings in one call, not one at a
    time, so bulk retrieval needs to preserve count and dimensionality
    correctly across every stored item."""
    embeddings = [make_random_embedding() for _ in range(5)]
    paths = [f"./data/images/img_{i}.jpg" for i in range(5)]
    temp_indexer.add_batch(embeddings, paths)

    stored = temp_indexer.collection.get(include=["embeddings", "metadatas"])

    assert len(stored["embeddings"]) == 5
    assert len(stored["metadatas"]) == 5
    assert all(len(vec) == EMBEDDING_DIM for vec in stored["embeddings"]), \
        "Every retrieved embedding should still be a valid 512-dim vector"