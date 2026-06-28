"""
__init__.py basically marks the `src` directory as a Python package

What this file does:

    Python treats any folder containing an '__init__.py' as a "package"
    Without it, you cannot do cross-file imports like:

        from src.embedder import CLIPEmbedder
        from src.indexer import VectorIndexer

    With it, Python knows `src` is a package and resolves those imports correctly

Why it can be empty:

    The mere presence of this file is enough to register the "package"
    Code inside '__init__.py' runs once when the package is first imported,
    so it's optionally used to expose a clean public API 
"""


# Optional: expose a clean public API for the package
# By importing here, it allows callers to write shorter import paths
#
# WITHOUT: 
#   from src.embedder import CLIPEmbedder
#   from src.indexer import VectorIndexer
#
# WITH:
#   from src import CLIPEmbedder, VectorIndexer
#
# This is purely convenience, for a small project like this it's optional, 
# but it's standard practice in larger libraries 

# from src.embedder import CLIPEmbedder
# from src.indexer import VectorIndexer