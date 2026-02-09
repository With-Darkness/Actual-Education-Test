"""Unit tests for the SAT Knowledge Matching System."""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base import KnowledgeBase
from src.embeddings import EmbeddingGenerator
from src.retrieval import RAGRetriever


class TestKnowledgeBase(unittest.TestCase):
    """Test KnowledgeBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        kb_path = Path(__file__).parent.parent / "data" / "sat_knowledge_base.json"
        self.kb = KnowledgeBase(str(kb_path))
    
    def test_load_knowledge_base(self):
        """Test that knowledge base loads correctly."""
        self.assertIsNotNone(self.kb.knowledge_points)
        self.assertGreater(len(self.kb.knowledge_points), 0)
    
    def test_get_all_points(self):
        """Test getting all knowledge points."""
        points = self.kb.get_all_points()
        self.assertIsInstance(points, list)
        self.assertGreater(len(points), 0)
    
    def test_get_by_category(self):
        """Test filtering by category."""
        math_points = self.kb.get_by_category("Math")
        self.assertGreater(len(math_points), 0)
        for point in math_points:
            self.assertEqual(point['category'].lower(), 'math')
    
    def test_get_by_id(self):
        """Test getting knowledge point by ID."""
        point = self.kb.get_by_id("MATH_001")
        self.assertIsNotNone(point)
        self.assertEqual(point['id'], "MATH_001")
    
    def test_get_text_for_embedding(self):
        """Test text generation for embedding."""
        point = self.kb.get_by_id("MATH_001")
        text = self.kb.get_text_for_embedding(point)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
    
    def test_get_statistics(self):
        """Test getting knowledge base statistics."""
        stats = self.kb.get_statistics()
        self.assertIn('total_points', stats)
        self.assertIn('categories', stats)
        self.assertGreater(stats['total_points'], 0)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test EmbeddingGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = EmbeddingGenerator()
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        text = "How do I solve quadratic equations?"
        embedding = self.embedder.encode_query(text)
        self.assertIsNotNone(embedding)
        self.assertGreater(len(embedding), 0)
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        texts = [
            "How do I solve quadratic equations?",
            "What is subject-verb agreement?"
        ]
        embeddings = self.embedder.encode(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))


class TestRAGRetriever(unittest.TestCase):
    """Test RAGRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        kb_path = Path(__file__).parent.parent / "data" / "sat_knowledge_base.json"
        self.kb = KnowledgeBase(str(kb_path))
        self.embedder = EmbeddingGenerator()
        self.retriever = RAGRetriever(self.kb, self.embedder)
    
    def test_retrieve(self):
        """Test basic retrieval."""
        query = "How do I solve quadratic equations?"
        results = self.retriever.retrieve(query, top_k=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for kp, score in results:
            self.assertIsInstance(kp, dict)
            self.assertIn('topic', kp)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_retrieve_relevance(self):
        """Test that retrieval returns relevant results."""
        query = "quadratic equations"
        results = self.retriever.retrieve(query, top_k=5)
        
        # Top result should be related to quadratic equations
        top_result = results[0][0]
        self.assertIn('quadratic', top_result.get('topic', '').lower() or 
                     top_result.get('description', '').lower())
    
    def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        results = self.retriever.retrieve("", top_k=5)
        self.assertEqual(len(results), 0)
    
    def test_retrieve_with_threshold(self):
        """Test retrieval with similarity threshold."""
        query = "How do I solve quadratic equations?"
        results = self.retriever.retrieve_with_threshold(
            query, top_k=5, min_similarity=0.3
        )
        
        # All results should meet threshold
        for _, score in results:
            self.assertGreaterEqual(score, 0.3)
    
    def test_get_index_stats(self):
        """Test getting index statistics."""
        stats = self.retriever.get_index_stats()
        self.assertIn('num_vectors', stats)
        self.assertIn('dimension', stats)
        self.assertGreater(stats['num_vectors'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        kb_path = Path(__file__).parent.parent / "data" / "sat_knowledge_base.json"
        self.kb = KnowledgeBase(str(kb_path))
        self.embedder = EmbeddingGenerator()
        self.retriever = RAGRetriever(self.kb, self.embedder)
    
    def test_end_to_end_search(self):
        """Test complete search workflow."""
        queries = [
            "quadratic equations",
            "subject-verb agreement",
            "Pythagorean theorem",
            "main idea",
            "context clues"
        ]
        
        for query in queries:
            results = self.retriever.retrieve(query, top_k=3)
            self.assertGreater(len(results), 0, f"No results for query: {query}")
            
            # Check that results are sorted by relevance
            scores = [score for _, score in results]
            self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
