"""Utility functions for the SAT knowledge matching system."""
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


def format_search_results(results: List[Tuple[Dict[str, Any], float]], 
                         format_type: str = "markdown") -> str:
    """
    Format search results for display.
    
    Args:
        results: List of (knowledge_point, similarity_score) tuples
        format_type: Output format ('markdown', 'text', 'json')
    
    Returns:
        Formatted string representation of results
    """
    if not results:
        return "No results found."
    
    if format_type == "json":
        return json.dumps([
            {
                "knowledge_point": kp,
                "similarity_score": float(score)
            }
            for kp, score in results
        ], indent=2)
    
    output_lines = []
    for i, (kp, similarity) in enumerate(results, 1):
        if format_type == "markdown":
            result_text = f"""
**Result {i}** (Relevance: {similarity:.2%})
- **Topic**: {kp.get('topic', 'N/A')}
- **Category**: {kp.get('category', 'N/A')} > {kp.get('subcategory', 'N/A')}
- **Description**: {kp.get('description', 'N/A')}
- **Key Concepts**: {', '.join(kp.get('key_concepts', [])[:3])}
- **Difficulty**: {kp.get('difficulty', 'N/A')}
"""
        else:  # text format
            result_text = f"""
Result {i} (Relevance: {similarity:.2%})
Topic: {kp.get('topic', 'N/A')}
Category: {kp.get('category', 'N/A')} > {kp.get('subcategory', 'N/A')}
Description: {kp.get('description', 'N/A')}
Key Concepts: {', '.join(kp.get('key_concepts', [])[:3])}
Difficulty: {kp.get('difficulty', 'N/A')}
"""
        output_lines.append(result_text)
    
    return "\n".join(output_lines)


def load_demo_cases(json_path: str) -> List[Dict[str, Any]]:
    """
    Load demo test cases from JSON file.
    
    Args:
        json_path: Path to demo cases JSON file
    
    Returns:
        List of demo case dictionaries
    """
    path = Path(json_path)
    if not path.exists():
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('demo_cases', [])


def save_demo_results(query: str, results: List[Tuple[Dict[str, Any], float]], 
                     output_path: str):
    """
    Save demo query results to a file.
    
    Args:
        query: Original query
        results: Search results
        output_path: Path to save results
    """
    output_data = {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "knowledge_point_id": kp.get('id'),
                "topic": kp.get('topic'),
                "category": kp.get('category'),
                "similarity_score": float(score)
            }
            for kp, score in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
