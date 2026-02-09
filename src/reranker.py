"""
BERT-based Reranker for RAG System

Uses cross-encoder models to rerank initial retrieval results.
Cross-encoders provide more accurate relevance scoring than bi-encoders
by jointly encoding query-document pairs.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder


class BERTReranker:
    """BERT-based reranker using cross-encoder models."""
    
    # Best reranker models (ranked by quality/speed balance)
    # ms-marco models are specifically trained for retrieval/reranking
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, good quality
    ALTERNATIVE_MODELS = {
        "high_quality": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Better quality, slower
        "general": "BAAI/bge-reranker-base",  # General purpose
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fastest
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize BERT reranker with cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model. If None, uses default.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"Loading reranker model: {self.model_name}...")
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            print("Reranker model loaded successfully!")
        except Exception as e:
            print(f"Error loading reranker model: {e}")
            print("Falling back to default model...")
            self.model_name = self.DEFAULT_MODEL
            self.model = CrossEncoder(self.model_name, max_length=512)
    
    def rerank(self, query: str, candidates: List[Tuple[Dict[str, Any], float]], 
               top_n: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidate knowledge points based on query relevance.
        
        Args:
            query: Student question or topic query
            candidates: List of (knowledge_point_dict, initial_score) tuples
            top_n: Number of top results to return after reranking. If None, returns all.
        
        Returns:
            List of reranked tuples: (knowledge_point_dict, rerank_score)
            Results are sorted by relevance (highest first)
        """
        if not candidates:
            return []
        
        if len(candidates) == 1:
            # No need to rerank a single candidate
            return candidates
        
        # Prepare query-document pairs for cross-encoder
        # Cross-encoder takes [query, document] pairs
        from .knowledge_base import KnowledgeBase
        
        # Create a temporary KB instance to format text (or use utility function)
        pairs = []
        for kp, _ in candidates:
            # Format knowledge point text for reranking
            doc_text = self._format_knowledge_point(kp)
            pairs.append([query, doc_text])
        
        # Get reranking scores from cross-encoder
        # Scores are relevance scores (higher = more relevant)
        scores = self.model.predict(pairs)
        
        # Convert to list if single score
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [float(scores)]
        
        # Combine candidates with rerank scores
        reranked_results = []
        for (kp, initial_score), rerank_score in zip(candidates, scores):
            # Normalize rerank score (cross-encoder scores can vary)
            # Most cross-encoders output scores in a range, normalize to 0-1
            normalized_score = self._normalize_score(rerank_score)
            
            # Optionally combine initial score and rerank score
            # Weighted combination: 70% rerank, 30% initial (rerank is more accurate)
            combined_score = 0.7 * normalized_score + 0.3 * initial_score
            
            reranked_results.append((kp, combined_score))
        
        # Sort by combined score (descending)
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n results
        if top_n is not None:
            return reranked_results[:top_n]
        
        return reranked_results
    
    def _format_knowledge_point(self, kp: Dict[str, Any]) -> str:
        """
        Format knowledge point into text for reranking.
        
        Args:
            kp: Knowledge point dictionary
        
        Returns:
            Formatted text string
        """
        parts = [
            f"Topic: {kp.get('topic', '')}",
            f"Description: {kp.get('description', '')}",
        ]
        
        # Add key concepts
        key_concepts = kp.get('key_concepts', [])
        if key_concepts:
            concepts_str = ", ".join(key_concepts[:5])  # Limit to avoid too long
            parts.append(f"Key Concepts: {concepts_str}")
        
        # Add category information
        category = kp.get('category', '')
        subcategory = kp.get('subcategory', '')
        if category:
            parts.append(f"Category: {category} - {subcategory}")
        
        return " | ".join(parts)
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize rerank score to 0-1 range.
        
        Different models output scores in different ranges.
        This function normalizes to a consistent 0-1 scale.
        
        Args:
            score: Raw rerank score
        
        Returns:
            Normalized score (0-1)
        """
        # Cross-encoder scores are typically in a range
        # Using sigmoid normalization for robustness
        # This works well for most cross-encoder models
        import math
        normalized = 1 / (1 + math.exp(-score))
        return max(0.0, min(1.0, normalized))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "cross-encoder",
            "max_length": 512,
            "description": "BERT-based cross-encoder for reranking retrieval results"
        }
