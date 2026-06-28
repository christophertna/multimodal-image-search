"""
indexer.py - ChromaDB vector store wrapper for persisting and retrieving CLIP embeddings

Architecture note:
    indexer.py has 1 responsibility: store and retrieve embedding vectors
    (It never actually touches the CLIP model)

    ChromaDB is a local-first vector database, it stores embeddings on disk
    and supports cosine similarity search natively. 
    
    Collections are analogous to tables in a relational DB: 
    
    each collection holds vectors of the same dimensionality plus their metadata

    Data flow:
        embedder.py  →  512-dim vector  →  indexer.add()  →  ChromaDB on disk
        text query   →  512-dim vector  →  indexer.search() →  top-k image paths
"""

import uuid
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
import torch


# Constants
EMBEDDING_DIM = 512                      # CLIP output dimension
DEFAULT_COLLECTION = "clip_image_index"  # rename
DEFAULT_TOP_K = 5                        # top 5 most similar results


# VectorIndexer class
class VectorIndexer:
    """
    Wraps ChromaDB, exposing three public methods:

        - add(embedding, image_path, metadata)  → stores a vector
        - add_batch(embeddings, image_paths, metadatas) → stores many at once
        - search(query_vector, top_k, filter)   → returns top-k (5) similar results
    """

    def __init__(self, persist_dir: str = "./index", collection_name: str = DEFAULT_COLLECTION):
        """
        Initialize ChromaDB in persistent mode and load (or create) a collection.

        Takes in 2 things:

            1- persist_dir:      Path on disk where ChromaDB will save its data.
                                 Relative to wherever you run the script from

            2- collection_name:  Name of the ChromaDB collection to use
        """

        # connect ChromaDB client to local disk (persistence)
        # (ff the directory doesn't exist yet, ChromaDB creates it automatically)

        # On restart, passing the same path reloads all previously stored vectors
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

        # Delegate collection setup to a private method (strucutre & clarity purposes)
        self.collection = self._init_collection(collection_name)

        print(f"[VectorIndexer] Collection '{collection_name}' ready — "
              f"{self.collection.count()} vectors stored.")


    # Collection management (private method from earlier)
    def _init_collection(self, name: str):
        """
        Load an existing collection or create a new one if it doesn't exist

        Why get_or_create_collection instead of create_collection?
            "create_collection" raises an error if the collection ALREADY exists, whereas
            "get_or_create_collection" is much safer & smarter choice to call on every startup/run (idempotency)

        Why cosine space?
            Recall: the embeddings are L2-normalized in embedder.py (unit vectors), so
            ChromaDB's cosine space computes: similarity = (a ⋅ b) / (||a|| ||b||) and

            since ||a|| = ||b|| = 1 after normalization, it just simplifies to (a ⋅ b)
            
            Cosine similarity ranges from -1 (complete opposite) to 1 (identical)
        """
        # get_or_create_collection takes a name and optional metadata:
        # "hnsw:space" metadata key tells ChromaDB which distance/similarity metric to use, so cosine here
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"} # cosine similarity
        )

    # now need a method to generate IDs and hash the image paths
    # The primary reason we hash the image path to generate IDs is to ensure IDEMPOTENCY: 
    # the ability to run the indexing process multiple times without creating duplicate data

    # If random ID (like a UUID) or an auto-incrementing counter were used instead for every image, 
    # running the indexing script twice on the same folder would result in the database storing the exact same image many times.

    # By using the image path as the basis for the ID, the system creates a "fingerprint"/record unique to that file location/image

    # ID generation helpers
    @staticmethod
    def _path_to_id(image_path: str) -> str:
        """
        Derive a stable & unique ID from a file path using MD5 hashing

        Why specifically hashing instead of a typical counter or UUID?
            - Hashing the path gives same ID for the same file, so re-indexing the same image is idempotent: 
              ChromaDB will update the existing entry instead of creating duplicates

            - A plain counter would generate a NEW id on every run (duplicate data)
        """
        # hashlib.md5() creates an MD5 hash object, then encode() converts the string to bytes (required by hashlib)
        # finally hexdigest() returns the hash as a 32-character hex string
        return hashlib.md5(image_path.encode()).hexdigest()
    
    # but why hash the file path at all in the first place?
    # - since file paths can vary in length, hashing them will produce a fixed lenght string for all paths (efficiency for database indexing)
    # - safety 
    

    @staticmethod
    def _random_id() -> str:
        """Generate a random UUID string (useful when no file path is available)"""

        # uuid.uuid4() generates a random 128-bit UUID
        # str() converts it to the standard "xxxxxxxx-xxxx-..." string format
        return str(uuid.uuid4())


    # Vector validation
    def _validate_embedding(self, embedding: torch.Tensor) -> list[float]:
        """
        Validate a tensor's shape and convert it to a plain Python list, since ChromaDB doesnt accept 
        PyTorch tensors since it expects a Python list of floats 
        
        Method also guards against dimension mismatches early, rather than letting ChromaDB throw an error

        
        Takes an embedding: 1D torch.Tensor, expected shape (512,) and returns a Python list of 512 floats

        Raise "ValueError" if the tensor is not 1D or has the wrong dimension        
        """

        # tensor.ndim returns the number of dimensions of the tensor, so a 1D tensor has ndim == 1 and 
        # a batch tensor like (1, 512) has ndim == 2
        if embedding.ndim != 1:
            raise ValueError(
                f"Expected a 1D embedding tensor, got shape {embedding.shape}. "
                "Did you forget to index with [0] after the model forward pass?"
            )

        # Check if embedding.shape[0] == EMBEDDING_DIM, if not then raise a ValueError 
        if embedding.shape[0] != EMBEDDING_DIM: # tensor.shape[0] gives the size of the first & only dimension of the tensor
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {embedding.shape[0]}. Are you using the right CLIP model?"
            )

        # finally .tolist() to convert the PyTorch tensor to a plain Python list
        return embedding.tolist()
        # recall: ChromaDB's Python client requires lists, not tensors (or numpy arrays)


    # Insertion (SINGLE)
    def add(
        self,
        embedding: torch.Tensor,
        image_path: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a single image embedding in the collection

        Args:
            embedding:   1D tensor of shape (512,) from embedder.embed_image().
            image_path:  Path to the source image — stored as metadata and used
                         to generate a stable ID.
            metadata:    Optional extra metadata dict (e.g. {"category": "nature"}).
                         The image_path is always included automatically.

        Returns:
            The generated document ID (MD5 of the path).
        """
        vector = self._validate_embedding(embedding)
        doc_id = self._path_to_id(image_path)

        # Merge caller-supplied metadata with the mandatory image_path field.
        # `{**a, **b}` merges two dicts; keys in b overwrite keys in a.
        combined_meta = {"image_path": image_path, **(metadata or {})}

        # collection.upsert() inserts the document if the ID is new,
        # or updates it if the ID already exists — making re-indexing safe.
        # All arguments are lists even for a single item (ChromaDB's API is
        # batch-first by design).


        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[combined_meta],
        )

        return doc_id


    # Insertion (BATCH)
    def add_batch(
        self,
        embeddings: list[torch.Tensor],
        image_paths: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        """
        Store multiple image embeddings in a single database call.

        Why batch insertion matters:
            Inserting N items one at a time means N round-trips to the DB.
            A single batch call sends all N items at once — much faster for
            large image folders.

        Args:
            embeddings:   List of 1D tensors, one per image.
            image_paths:  Corresponding list of image file paths.
            metadatas:    Optional list of metadata dicts, one per image.
                          If omitted, only image_path is stored per entry.

        Returns:
            List of generated document IDs (one MD5 hash per path).
        """
        if len(embeddings) != len(image_paths):
            raise ValueError(
                f"embeddings and image_paths must have the same length "
                f"({len(embeddings)} vs {len(image_paths)})."
            )

        # Build parallel lists that ChromaDB's batch API expects.
        vectors = [self._validate_embedding(e) for e in embeddings]
        ids     = [self._path_to_id(p) for p in image_paths]

        # Merge image_path into each metadata dict (or create one if absent).
        if metadatas is None:
            metadatas = [{}] * len(image_paths)
        combined_metas = [
            {"image_path": path, **meta}
            for path, meta in zip(image_paths, metadatas)
        ]

        # same method as add(), just pass the full lists directly.
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=combined_metas,
        )

        return ids


    # Retrieval
    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = DEFAULT_TOP_K,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Find the top-k most similar stored embeddings to a query vector.

        Args:
            query_vector:  1D tensor of shape (512,) from embedder.embed_text()
                           or embedder.embed_image().
            top_k:         Number of results to return.
            filter:        Optional ChromaDB metadata filter dict.
                           Example: {"category": {"$eq": "nature"}}
                           Only documents matching the filter are searched.

        Returns:
            A list of dicts, each containing:
                {
                    "id":        str,   # document ID
                    "image_path": str,  # path to the matched image
                    "distance":  float, # cosine distance (lower = more similar)
                    "metadata":  dict,  # full metadata dict stored at index time
                }
        """
        # Guard: searching an empty collection raises a confusing ChromaDB error.
        if self.collection.count() == 0:
            raise RuntimeError(
                "The collection is empty. Index some images first with add() or add_batch()."
            )

        vector = self._validate_embedding(query_vector)

        # collection.query() is ChromaDB's similarity search method
        # `query_embeddings` is a list of query vectors (pass one)

        # `n_results` is top-k
        # `where` is the optional metadata filter dict
        # `include` controls which fields are returned in the response
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter,
            include=["metadatas", "distances"],
        )

        # ChromaDB returns results batched by query, so results["ids"] is
        # a list of lists: [[id1, id2, ...]]. We queried with one vector, so
        # we take [0] to get the flat list for our single query
        ids        = results["ids"][0]
        distances  = results["distances"][0]
        metadatas  = results["metadatas"][0]

        # Zip ids, distances, and metadatas together into a list of
        # dicts with keys: "id", "image_path", "distance", "metadata"
        return [
            {
                "id": doc_id,
                "image_path": meta.get("image_path", ""),
                "distance": dist,
                "metadata": meta,
            }
            for doc_id, dist, meta in zip(ids, distances, metadatas)
        ]

 
    # Utility
    def count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection, wiping all stored vectors

        Use with caution — this is irreversible without re-indexing
        Useful during development when you want a clean slate
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self._init_collection(self.collection_name)
        print(f"[VectorIndexer] Collection '{self.collection_name}' has been reset.")