"""Knowledge base loader and manager for SAT curriculum content."""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class KnowledgeBase:
    """Manages SAT knowledge base loading and querying."""
    
    def __init__(self, json_path: str):
        """
        Initialize knowledge base from JSON file.
        
        Args:
            json_path: Path to JSON file containing knowledge points
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.knowledge_points = self.data.get('knowledge_points', [])
        if not self.knowledge_points:
            raise ValueError("Knowledge base is empty or invalid format")
    
    def get_all_points(self) -> List[Dict[str, Any]]:
        """
        Return all knowledge points.
        
        Returns:
            List of all knowledge point dictionaries
        """
        return self.knowledge_points
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Filter knowledge points by category.
        
        Args:
            category: Category name (e.g., 'Math', 'Reading', 'Writing')
        
        Returns:
            List of knowledge points in the specified category
        """
        return [kp for kp in self.knowledge_points 
                if kp.get('category', '').lower() == category.lower()]
    
    def get_by_subcategory(self, subcategory: str) -> List[Dict[str, Any]]:
        """
        Filter knowledge points by subcategory.
        
        Args:
            subcategory: Subcategory name (e.g., 'Algebra', 'Geometry')
        
        Returns:
            List of knowledge points in the specified subcategory
        """
        return [kp for kp in self.knowledge_points 
                if kp.get('subcategory', '').lower() == subcategory.lower()]
    
    def get_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific knowledge point by ID.
        
        Args:
            point_id: Knowledge point ID (e.g., 'MATH_001')
        
        Returns:
            Knowledge point dictionary or None if not found
        """
        for kp in self.knowledge_points:
            if kp.get('id') == point_id:
                return kp
        return None
    
    def get_text_for_embedding(self, knowledge_point: Dict[str, Any]) -> str:
        """
        Convert knowledge point to text for embedding generation.
        Combines multiple fields to create a comprehensive text representation.
        
        Args:
            knowledge_point: Knowledge point dictionary
        
        Returns:
            Combined text string for embedding
        """
        parts = [
            knowledge_point.get('topic', ''),
            knowledge_point.get('description', ''),
            ' '.join(knowledge_point.get('key_concepts', [])),
            ' '.join(knowledge_point.get('common_applications', []))
        ]
        return ' '.join(parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics (total points, categories, etc.)
        """
        categories = {}
        subcategories = {}
        
        for kp in self.knowledge_points:
            cat = kp.get('category', 'Unknown')
            subcat = kp.get('subcategory', 'Unknown')
            
            categories[cat] = categories.get(cat, 0) + 1
            subcategories[subcat] = subcategories.get(subcat, 0) + 1
        
        return {
            'total_points': len(self.knowledge_points),
            'categories': categories,
            'subcategories': subcategories
        }
