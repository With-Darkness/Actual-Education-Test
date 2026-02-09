"""Embedding generation using sentence transformers."""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingGenerator:
    """Generates embeddings for text using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence transformer model.
                       'all-MiniLM-L6-v2' is fast and efficient for semantic search.
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing (larger = faster but more memory)
        
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10
        )
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text string
        
        Returns:
            Numpy array of embedding (1D array)
        """
        return self.model.encode([query], convert_to_numpy=True)[0]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        # Get dimension by encoding a dummy text
        dummy_embedding = self.encode_query("test")
        return len(dummy_embedding)
