"""SAT Knowledge Matching System - Core modules."""
from .knowledge_base import KnowledgeBase
from .embeddings import EmbeddingGenerator
from .retrieval import RAGRetriever
from .reranker import BERTReranker
from .query_enrichment import QueryEnricher

try:
    from .evaluation import RAGEvaluator, SimpleEvaluator
    __all__ = ['KnowledgeBase', 'EmbeddingGenerator', 'RAGRetriever', 'BERTReranker', 
               'QueryEnricher', 'RAGEvaluator', 'SimpleEvaluator']
except ImportError:
    __all__ = ['KnowledgeBase', 'EmbeddingGenerator', 'RAGRetriever', 'BERTReranker', 'QueryEnricher']
