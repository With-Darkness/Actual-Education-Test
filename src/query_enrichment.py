"""
Query Enrichment Module

Enriches student questions before retrieval to improve RAG system performance.
Supports multiple enrichment strategies: expansion, rewriting, and LLM-based enhancement.
"""
from typing import List, Dict, Any, Optional
import re


class QueryEnricher:
    """Enriches queries to improve retrieval quality."""
    
    # Common SAT-related synonyms and expansions
    SAT_SYNONYMS = {
        "solve": ["find solution", "calculate", "work out", "determine"],
        "explain": ["describe", "clarify", "elaborate", "detail"],
        "understand": ["comprehend", "grasp", "learn", "master"],
        "how": ["what is the method", "what is the process", "steps to"],
        "what": ["define", "explain", "describe"],
        "formula": ["equation", "expression", "rule"],
        "rule": ["principle", "concept", "guideline"],
        "problem": ["question", "exercise", "example"],
        "graph": ["plot", "chart", "diagram"],
        "function": ["equation", "relationship", "mapping"]
    }
    
    # Common SAT topic expansions
    TOPIC_EXPANSIONS = {
        "quadratic": ["parabola", "second degree", "x squared"],
        "linear": ["straight line", "first degree", "proportional"],
        "equation": ["formula", "expression", "relationship"],
        "inequality": ["comparison", "greater than", "less than"],
        "triangle": ["three-sided", "polygon", "geometry"],
        "circle": ["round", "circular", "radius", "diameter"],
        "grammar": ["syntax", "sentence structure", "language rules"],
        "vocabulary": ["word meaning", "definitions", "terms"],
        "reading": ["comprehension", "passage analysis", "text understanding"],
        "writing": ["composition", "essay", "grammar"]
    }
    
    def __init__(self, enable_expansion: bool = True, enable_rewriting: bool = True,
                 enable_llm_enhancement: bool = False, llm_model: Optional[Any] = None):
        """
        Initialize query enricher.
        
        Args:
            enable_expansion: Enable synonym/term expansion
            enable_rewriting: Enable query rewriting strategies
            enable_llm_enhancement: Enable LLM-based enhancement (requires model)
            llm_model: Optional LLM model for advanced enrichment
        """
        self.enable_expansion = enable_expansion
        self.enable_rewriting = enable_rewriting
        self.enable_llm_enhancement = enable_llm_enhancement
        self.llm_model = llm_model
    
    def enrich(self, query: str, strategy: str = "auto") -> str:
        """
        Enrich a query using specified strategy.
        
        Args:
            query: Original student question
            strategy: Enrichment strategy - "auto", "expansion", "rewriting", "llm", or "none"
        
        Returns:
            Enriched query string
        """
        if not query or not query.strip():
            return query
        
        query = query.strip()
        
        if strategy == "none":
            return query
        
        if strategy == "auto":
            # Use expansion and rewriting by default
            enriched = self._apply_expansion(query)
            enriched = self._apply_rewriting(enriched)
            return enriched
        
        if strategy == "expansion":
            return self._apply_expansion(query)
        
        if strategy == "rewriting":
            return self._apply_rewriting(query)
        
        if strategy == "llm":
            if self.enable_llm_enhancement and self.llm_model:
                return self._apply_llm_enhancement(query)
            else:
                # Fallback to rewriting if LLM not available
                return self._apply_rewriting(query)
        
        # Default: apply all available strategies
        enriched = query
        if self.enable_expansion:
            enriched = self._apply_expansion(enriched)
        if self.enable_rewriting:
            enriched = self._apply_rewriting(enriched)
        if self.enable_llm_enhancement and self.llm_model:
            enriched = self._apply_llm_enhancement(enriched)
        
        return enriched
    
    def _apply_expansion(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
        
        Returns:
            Expanded query
        """
        if not self.enable_expansion:
            return query
        
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            # Check for synonyms
            if clean_word in self.SAT_SYNONYMS:
                synonyms = self.SAT_SYNONYMS[clean_word]
                expanded_terms.append(word)
                expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
            else:
                expanded_terms.append(word)
            
            # Check for topic expansions
            for topic, expansions in self.TOPIC_EXPANSIONS.items():
                if topic in clean_word:
                    expanded_terms.extend(expansions[:2])
                    break
        
        # Also add full phrase expansions
        query_lower = query.lower()
        for topic, expansions in self.TOPIC_EXPANSIONS.items():
            if topic in query_lower:
                expanded_terms.extend(expansions)
        
        # Combine original query with expansions
        if len(expanded_terms) > len(words):
            # Add expansions to original query
            enriched = query + " " + " ".join(set(expanded_terms) - set(words))
            return enriched
        
        return query
    
    def _apply_rewriting(self, query: str) -> str:
        """
        Rewrite query to improve retrieval.
        
        Args:
            query: Original query
        
        Returns:
            Rewritten query
        """
        if not self.enable_rewriting:
            return query
        
        rewritten = query
        
        # Add context for common question patterns
        patterns = [
            (r"how (do|can) i (.+)", r"how to \2, steps for \2, method to \2"),
            (r"what is (.+)", r"what is \1, definition of \1, explain \1"),
            (r"how to (.+)", r"how to \1, steps to \1, process of \1"),
            (r"explain (.+)", r"explain \1, describe \1, what is \1"),
            (r"(.+)\?", r"\1, concept of \1, understanding \1"),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                # Add related phrases
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    captured = match.group(2) if match.lastindex >= 2 else match.group(1)
                    related = replacement.replace(r"\1", captured).replace(r"\2", captured)
                    rewritten = f"{query} {related}"
                    break
        
        # Add SAT-specific context
        sat_keywords = ["sat", "test", "exam", "question", "problem"]
        if any(keyword in query.lower() for keyword in sat_keywords):
            # Already has SAT context
            pass
        else:
            # Add educational context for academic queries
            academic_patterns = [
                r"(solve|find|calculate|determine)",
                r"(equation|formula|expression)",
                r"(concept|topic|subject)"
            ]
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in academic_patterns):
                rewritten = f"{query} SAT math concept"
        
        return rewritten.strip()
    
    def _apply_llm_enhancement(self, query: str) -> str:
        """
        Use LLM to enhance query (if available).
        
        Args:
            query: Original query
        
        Returns:
            LLM-enhanced query
        """
        if not self.enable_llm_enhancement or not self.llm_model:
            return query
        
        # Placeholder for LLM enhancement
        # In production, this would use an LLM to:
        # 1. Expand the query with related concepts
        # 2. Rewrite for better retrieval
        # 3. Add context based on domain knowledge
        
        # Example prompt:
        # "Enhance this student question for better retrieval in an SAT knowledge base: {query}"
        
        # For now, fallback to rewriting
        return self._apply_rewriting(query)
    
    def enrich_multiple(self, queries: List[str], strategy: str = "auto") -> List[str]:
        """
        Enrich multiple queries.
        
        Args:
            queries: List of queries
            strategy: Enrichment strategy
        
        Returns:
            List of enriched queries
        """
        return [self.enrich(q, strategy) for q in queries]
    
    def get_enrichment_info(self) -> Dict[str, Any]:
        """
        Get information about enrichment capabilities.
        
        Returns:
            Dictionary with enrichment info
        """
        return {
            "expansion_enabled": self.enable_expansion,
            "rewriting_enabled": self.enable_rewriting,
            "llm_enhancement_enabled": self.enable_llm_enhancement,
            "llm_model_available": self.llm_model is not None,
            "synonyms_count": len(self.SAT_SYNONYMS),
            "topic_expansions_count": len(self.TOPIC_EXPANSIONS)
        }


class SimpleQueryEnricher:
    """
    Simplified query enricher for basic use cases.
    Lightweight version without LLM dependencies.
    """
    
    @staticmethod
    def enrich(query: str) -> str:
        """
        Simple query enrichment.
        
        Args:
            query: Original query
        
        Returns:
            Enriched query
        """
        if not query or not query.strip():
            return query
        
        query = query.strip()
        
        # Add common SAT context
        sat_contexts = {
            "how": "how to solve",
            "what": "what is",
            "explain": "explain concept",
            "solve": "solve problem"
        }
        
        # Check if query already has good structure
        query_lower = query.lower()
        
        # Add related terms for common patterns
        if "?" in query:
            # Remove question mark and add variations
            base = query.rstrip("?")
            enriched = f"{base} concept method approach"
        else:
            enriched = f"{query} concept method"
        
        return enriched.strip()
