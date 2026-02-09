"""RAG-based retrieval system using FAISS for vector similarity search with BERT reranking."""
import faiss
import numpy as np
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .knowledge_base import KnowledgeBase
from .embeddings import EmbeddingGenerator
from .reranker import BERTReranker
from .query_enrichment import QueryEnricher


class RAGRetriever:
    """Retrieval-Augmented Generation system for SAT knowledge matching."""
    
    def __init__(self, knowledge_base: KnowledgeBase, embedding_model: EmbeddingGenerator,
                 reranker: Optional[BERTReranker] = None,
                 query_enricher: Optional[QueryEnricher] = None,
                 index_dir: Optional[str] = None):
        """
        Initialize RAG retriever with knowledge base and embedding model.
        
        Args:
            knowledge_base: KnowledgeBase instance
            embedding_model: EmbeddingGenerator instance
            reranker: Optional BERTReranker instance for two-stage retrieval
            query_enricher: Optional QueryEnricher instance for query enhancement
            index_dir: Directory to store/load FAISS index. If None, uses 'data/faiss_index'
        """
        self.kb = knowledge_base
        self.embedder = embedding_model
        self.reranker = reranker
        self.query_enricher = query_enricher
        self.index: Optional[faiss.Index] = None
        self.knowledge_points: List[Dict[str, Any]] = []
        
        # Set up index directory
        if index_dir is None:
            index_dir = Path(__file__).parent.parent / "data" / "faiss_index"
        else:
            index_dir = Path(index_dir)
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.index_dir / "index.bin"
        self.metadata_path = self.index_dir / "metadata.json"
        
        # Try to load existing index, otherwise build new one
        if not self._load_index():
            self._build_index()
    
    def _get_knowledge_base_hash(self) -> str:
        """
        Generate a hash of the knowledge base to detect changes.
        
        Returns:
            SHA256 hash string of knowledge base content
        """
        knowledge_points = self.kb.get_all_points()
        
        # Create a stable representation for hashing
        kb_data = {
            "count": len(knowledge_points),
            "ids": sorted([kp.get('id', '') for kp in knowledge_points]),
            "kb_path": str(self.kb.json_path),
            "kb_mtime": self.kb.json_path.stat().st_mtime if self.kb.json_path.exists() else 0
        }
        
        kb_string = json.dumps(kb_data, sort_keys=True)
        return hashlib.sha256(kb_string.encode()).hexdigest()
    
    def _load_index(self) -> bool:
        """
        Load FAISS index from disk if it exists and is up-to-date.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            print("No existing index found, will build new index...")
            return False
        
        try:
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if knowledge base has changed
            current_hash = self._get_knowledge_base_hash()
            stored_hash = metadata.get('knowledge_base_hash', '')
            
            # Check embedding model compatibility
            stored_model = metadata.get('embedding_model', '')
            current_model = self.embedder.model_name
            
            if current_hash != stored_hash:
                print(f"Knowledge base has changed (hash mismatch), rebuilding index...")
                return False
            
            if stored_model != current_model:
                print(f"Embedding model changed ({stored_model} -> {current_model}), rebuilding index...")
                return False
            
            # Load FAISS index
            print(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(str(self.index_path))
            
            # Load knowledge points (they should match the index)
            knowledge_points = self.kb.get_all_points()
            self.knowledge_points = knowledge_points
            
            # Verify index matches knowledge base
            if self.index.ntotal != len(knowledge_points):
                print(f"Index size mismatch ({self.index.ntotal} vs {len(knowledge_points)}), rebuilding...")
                return False
            
            print(f"Index loaded successfully! ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Will rebuild index...")
            return False
    
    def _build_index(self):
        """Build FAISS index from knowledge base embeddings and save to disk."""
        print("Building FAISS index...")
        knowledge_points = self.kb.get_all_points()
        self.knowledge_points = knowledge_points
        
        # Generate embeddings for all knowledge points
        texts = [self.kb.get_text_for_embedding(kp) for kp in knowledge_points]
        embeddings = self.embedder.encode(texts)
        
        # Create FAISS index (L2 distance for similarity search)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity (more accurate for semantic search)
        # L2 normalization allows using L2 distance as cosine distance
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save index to disk
        self._save_index()
        
        print(f"Index built and saved successfully with {len(knowledge_points)} knowledge points!")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            print(f"Saving FAISS index to {self.index_path}...")
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            metadata = {
                "knowledge_base_hash": self._get_knowledge_base_hash(),
                "embedding_model": self.embedder.model_name,
                "num_vectors": self.index.ntotal,
                "dimension": self.index.d,
                "index_type": type(self.index).__name__,
                "knowledge_base_path": str(self.kb.json_path)
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Index metadata saved to {self.metadata_path}")
            
        except Exception as e:
            print(f"Warning: Could not save index to disk: {e}")
            print("Index will be rebuilt on next startup")
    
    def update_index(self):
        """
        Force rebuild and update the index.
        Useful when knowledge base is modified programmatically.
        """
        print("Updating index...")
        self._build_index()
    
    def retrieve(self, query: str, top_k: int = 5, enrich_query: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most relevant knowledge points for a query.
        
        Args:
            query: Student question or topic query
            top_k: Number of top results to return
            enrich_query: Whether to enrich the query before retrieval
        
        Returns:
            List of tuples: (knowledge_point_dict, similarity_score)
            Results are sorted by relevance (highest first)
        """
        if not query or not query.strip():
            return []
        
        # Enrich query if enricher is available and enabled
        original_query = query
        if enrich_query and self.query_enricher:
            query = self.query_enricher.enrich(query, strategy="auto")
        
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        # We use top_k + 1 to handle potential duplicates, then take top_k
        k = min(top_k, len(self.knowledge_points))
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert distances to similarity scores
        # Since we're using normalized vectors, distance is cosine distance
        # Similarity = 1 - normalized_distance (closer to 0 distance = higher similarity)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # Convert cosine distance to similarity score (0-1 scale)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # Similarity = 1 - (distance / 2) gives us 0-1 scale
            similarity = max(0, 1 - (dist / 2))
            results.append((self.knowledge_points[idx], similarity))
        
        # Results are already sorted by distance (lowest first = highest similarity)
        return results
    
    def retrieve_with_threshold(self, query: str, top_k: int = 5, 
                                min_similarity: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve knowledge points with a minimum similarity threshold.
        
        Args:
            query: Student question or topic query
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity score threshold (0-1)
        
        Returns:
            List of tuples: (knowledge_point_dict, similarity_score)
            Filtered by similarity threshold
        """
        results = self.retrieve(query, top_k * 2)  # Get more to filter
        filtered = [(kp, score) for kp, score in results if score >= min_similarity]
        return filtered[:top_k]
    
    def retrieve_with_reranking(self, query: str, m: int = 20, n: int = 5, 
                                enrich_query: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Two-stage retrieval: retrieve m candidates, rerank with BERT, return top n.
        
        This method implements a two-stage retrieval pipeline:
        1. Query Enrichment (optional): Enhance query for better retrieval
        2. Stage 1 (FAISS): Fast semantic search to retrieve m candidates
        3. Stage 2 (BERT Reranker): Accurate reranking of candidates to get top n
        
        Args:
            query: Student question or topic query
            m: Number of candidates to retrieve in first stage (FAISS)
            n: Number of final results to return after reranking
            enrich_query: Whether to enrich the query before retrieval
        
        Returns:
            List of tuples: (knowledge_point_dict, rerank_score)
            Results are sorted by relevance (highest first)
        """
        if not query or not query.strip():
            return []
        
        # Enrich query if enricher is available and enabled
        original_query = query
        if enrich_query and self.query_enricher:
            query = self.query_enricher.enrich(query, strategy="auto")
        
        if self.reranker is None:
            # Fallback to regular retrieval if reranker not available
            print("Warning: Reranker not initialized, using regular retrieval")
            return self.retrieve(query, top_k=n, enrich_query=False)  # Already enriched
        
        # Stage 1: Retrieve m candidates using FAISS (fast semantic search)
        candidates = self.retrieve(query, top_k=m, enrich_query=False)  # Already enriched
        
        if not candidates:
            return []
        
        # Stage 2: Rerank candidates using BERT cross-encoder (accurate scoring)
        reranked_results = self.reranker.rerank(query, candidates, top_n=n)
        
        return reranked_results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "Index not built"}
        
        stats = {
            "num_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "similarity_algorithm": "Cosine Similarity (L2 normalized)",
            "reranker_enabled": self.reranker is not None,
            "index_persisted": self.index_path.exists() if self.index_path else False
        }
        
        # Add metadata if available
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                stats["index_metadata"] = metadata
            except Exception:
                pass
        
        if self.reranker:
            stats["reranker_model"] = self.reranker.get_model_info()
        
        if self.query_enricher:
            stats["query_enrichment"] = self.query_enricher.get_enrichment_info()
        
        return stats