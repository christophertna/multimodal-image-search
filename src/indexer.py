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
        
        Also checks for dimension mismatches early, rather than letting ChromaDB throw an error

        
        Takes an embedding: 1D torch.Tensor, expected shape (512,) and returns a Python list of 512 floats

        Raise "ValueError" if the tensor is not 1D or has the wrong dimension        
        """

        # tensor.ndim returns the number of dimensions of the tensor, so a 1D tensor has ndim == 1 and 
        # a batch tensor like (1, 512) has ndim == 2
        if embedding.ndim != 1: # return error if ndmi isnt 1
            raise ValueError(
                f"Expected a 1D embedding tensor, got shape {embedding.shape}. "
                "Did you forget to index with [0] after the model forward pass?"
            )

        # Check if embedding.shape[0] == EMBEDDING_DIM, if not then raise a ValueError 
        if embedding.shape[0] != EMBEDDING_DIM: # tensor.shape[0] gives the size of the first & only dimension of the tensor
            raise ValueError( # return error if tensor isnt 512 dim
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {embedding.shape[0]}. Are you using the right CLIP model?"
            )

        # finally .tolist() to convert the PyTorch tensor to a plain Python list
        return embedding.tolist()
        # recall: ChromaDB's Python client requires lists, not tensors (or numpy arrays)


    # Insertion of a SINGLE IMAGE only (to the ChromaDB collection)
    def add(
        self,
        embedding: torch.Tensor,
        image_path: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a single image embedding in the collection

        Takes as input:

            embedding:   1D tensor (512,) from embedder.embed_image() (recall method from embedder.py)

            image_path:  Path to the source image (stored as metadata and used to generate a stable ID)

            metadata:    Optional extra metadata dict (e.g. {"category": "nature"}) & image_path is always included automatically

        And returns:

            Generated document ID (MD5/hash of the path)
        """
       
        vector = self._validate_embedding(embedding) # run the vector validation method from earlier to check
        doc_id = self._path_to_id(image_path) # once vector validated, hash the image path (using method from earlier)

        # Merge metadata with its corresponding image_path field to create a single dictionnary for the new image
        combined_meta = {"image_path": image_path, **(metadata or {})} 
        # notice the "metadata or {}"", so metadata is optional and can be none in the new dictionnary

        # collection.upsert() inserts the document if the ID is NEW or updates it if the ID already exists (re-indexing safe/no duplicates)
        # All arguments are LISTS even for a single item (ChromaDB's API is batch-first)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[combined_meta],
        )

        # basically the ingestion/storage phase (add method)

        # so upload image, embed image, normalize it, validate the embedding, hash image path, 
        # extracting any metadata, create new dictionnary object with image path and metadata,
        # insert enw dictionnary in ChromaDB collection, and return the doc id
        return doc_id


    # Insertion by BATCH of images (to ChromaDB collection)
    def add_batch(
        self,
        embeddings: list[torch.Tensor],
        image_paths: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        """
        Store multiple image embeddings in a single database call:

        Why batch insertion?

            Inserting N items one at a time --> N round-trips to DB (not efficient)

            A single batch call sends all N items at once so way faster for larger image folders 


        Takes in as input:

            embeddings:   List of 1D tensors, 1 per image

            image_paths:  Corresponding LIST of image file paths

            metadatas:    Optional list of metadata dicts, 1 per image (if none only image_path is stored per entry)
    
        And returns:

            List of generated document IDs (one MD5 hash per path)
        """

        # first ensure both lists are the same size, e.g. 1 tensor/image for every string/image file path
        if len(embeddings) != len(image_paths):
            raise ValueError(
                f"embeddings and image_paths must have the same length "
                f"({len(embeddings)} vs {len(image_paths)})."
            )

        # Build parallel lists that ChromaDB's batch API expects 
        vectors = [self._validate_embedding(e) for e in embeddings] # validate embedding for every tensor in 'embeddings' & store result in 'vectors'
        ids     = [self._path_to_id(p) for p in image_paths]        # generate hashes for every string in 'image_paths' & store result in 'ids'
        # (index 0 in 'vectors' corresponds to index 0 in 'ids')

        # Merge image_path into each metadata dict (or create one if absent)
        # similar to the single image process, but this version is a loop:
        # ex: if 100 images, loop generates 100 dictionnary objects (contains image_path & any associated metadata for that image)

        # first handle the case where there is no metadata
        if metadatas is None:
            metadatas = [{}] * len(image_paths) # creates a list of empty dictionaries so that the zip function has something to pair with each image_path
                                                # Even if meta is {}, dictionary will still correctly contain the image_path
        combined_metas = [
            {"image_path": path, **meta} # recall: ** extracts the dict values from the metadata dict
            for path, meta in zip(image_paths, metadatas) # actual loop code line
        ]
        # zip(image_paths, metadatas) is basically the main loop action:

        # Takes the 1st element from 'image_paths' and the 1st element from 'metadatas'
        # Hands them to the dictionary creation logic: {"image_path": path, meta}
        # Repeats this for every single index until the end of the lists

        # same method as single image 'add()', but just passes the full lists directly (already in lists)
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=combined_metas,
        )

        # return all new created ids
        return ids


    # Retrieval
    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = DEFAULT_TOP_K,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Find the top-k most similar stored embeddings to a given/input query vector

        Takes in:

            query_vector:  1D tensor (512,) from embedder.embed_text() or embedder.embed_image()

            top_k:         Number of results to return (constant set to 5)

            filter:        Optional ChromaDB metadata filter dict.
                           Example: {"category": {"$eq": "nature"}}
                           Only documents matching the filter are searched.

        And returns:

            A list of dicts, each containing:
                {
                    "id":        str,   # document ID
                    "image_path": str,  # path to the matched image
                    "distance":  float, # cosine distance (lower = more similar)
                    "metadata":  dict,  # full metadata dict stored at index time
                }
        """

        # Searching an empty collection raises a ChromaDB error,
        # so handle the empty collection case first
        if self.collection.count() == 0:
            raise RuntimeError(
                "The collection is empty. Index some images first with add() or add_batch()."
            )

        # validate the query/input vector 
        vector = self._validate_embedding(query_vector)

        # collection.query() is ChromaDB's similarity search method 
        results = self.collection.query(
            query_embeddings=[vector],              # "query_embeddings" is a list of query vectors (pass 1 vector here) since ChromaDB expects lists/batches
            n_results=top_k,                        # "n_results" is top-k constant --> (5)
            where=filter,                           # metadata acts as a "where" filter in the collection
            include=["metadatas", "distances"],     # "include" controls which fields are returned in the response later
        ) 
        # remember, its only passing a list containing a single vector, the query vector + the other stuff

        # ChromaDB returns results batched by query
        # We queried with 1 vector so we take its first element [0] to get the flat list for our single query
        ids        = results["ids"][0]       
        distances  = results["distances"][0]
        metadatas  = results["metadatas"][0]

        # results["ids"] is a list of lists: [[id1, id2, ...]] (outer list is a SINGLE list containing the list of ids), so need [0], 
        # to get: ["id1", "id2", "id3"] --> (flat list of IDs)

        # now we have 3 parallel lists extracted from the collection

        # Zip ids, distances, and metadatas together into a list of dicts with keys: "id", "image_path", "distance", "metadata"
        return [
            {
                "id": doc_id,
                "image_path": meta.get("image_path", ""),
                "distance": dist, # from cosine similarity
                "metadata": meta,
            }
            for doc_id, dist, meta in zip(ids, distances, metadatas) # main loop action
        ]
        # loop action:
        # It takes one doc_id, one dist, and one meta
        # Packs them into a brand-new dictionary
        # Places that dictionary into the final list that will be returned to whoever called the search() function
 
 
    # Other utility stuff
    def count(self) -> int:
        """Return number of vectors currently stored in the collection"""

        return self.collection.count()


    # data/collection reset
    def reset(self) -> None:
        """
        Delete and recreate the collection, wiping all stored vectors

        DANGEROUS: irreversible without re-indexing, but useful during development
        """

        self.client.delete_collection(self.collection_name)
        self.collection = self._init_collection(self.collection_name)
        print(f"[VectorIndexer] Collection '{self.collection_name}' has been reset.")