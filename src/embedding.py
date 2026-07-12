import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity



"""
This module provides functionality for generating embeddings for text documents using the SentenceTransformer model.
 It includes the EmbeddingManager class, which handles loading the model and generating embeddings for a list of texts.

"""

class EmbeddingManager:
    
    """Load a sentence-transformer model and generate embeddings for text."""
    
    def __init__ (self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model once during initialization."""
        try:

            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")

        except Exception as e:
            
            print(f"Error loading model {self.model_name}: {e}")
            raise e

    def generate_embeddings (self, texts: List[str]) -> np.ndarray:

        """Convert a list of text strings into embedding vectors."""

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
