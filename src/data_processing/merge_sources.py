"""
Data Merger Script

Merges knowledge points from multiple sources into a single unified knowledge base.
Handles duplicate detection, ID conflicts, and data validation.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
import re


class KnowledgeBaseMerger:
    """Merges knowledge points from multiple sources."""
    
    def __init__(self, processed_dir: str = "data/processed", 
                 output_path: str = "data/sat_knowledge_base_expanded.json"):
        """
        Initialize merger.
        
        Args:
            processed_dir: Directory containing source JSON files
            output_path: Path to save merged knowledge base
        """
        self.processed_dir = Path(processed_dir)
        self.output_path = Path(output_path)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_source_files(self) -> List[Dict[str, Any]]:
        """
        Load all source JSON files from processed directory.
        
        Returns:
            List of source data dictionaries
        """
        sources = []
        source_files = [
            "khan_academy_sat_content.json",
            "college_board_sat_content.json",
            "educational_sat_content.json"
        ]
        
        for filename in source_files:
            file_path = self.processed_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sources.append(data)
                        print(f"Loaded {len(data.get('knowledge_points', []))} points from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: {filename} not found, skipping...")
        
        return sources
    
    def detect_duplicates(self, knowledge_points: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Detect duplicate knowledge points based on topic similarity.
        
        Args:
            knowledge_points: List of knowledge point dictionaries
        
        Returns:
            Dictionary mapping topic to list of indices
        """
        topic_to_indices = defaultdict(list)
        
        for idx, kp in enumerate(knowledge_points):
            topic = kp.get('topic', '').lower().strip()
            topic_to_indices[topic].append(idx)
        
        # Find duplicates (topics that appear multiple times)
        duplicates = {topic: indices for topic, indices in topic_to_indices.items() 
                     if len(indices) > 1}
        
        return duplicates
    
    def merge_duplicate(self, knowledge_points: List[Dict[str, Any]], 
                        indices: List[int]) -> Dict[str, Any]:
        """
        Merge duplicate knowledge points, combining information from all sources.
        
        Args:
            knowledge_points: List of all knowledge points
            indices: Indices of duplicate entries
        
        Returns:
            Merged knowledge point dictionary
        """
        # Use the first one as base
        merged = knowledge_points[indices[0]].copy()
        
        # Combine information from all duplicates
        all_sources = set()
        all_key_concepts = set()
        all_applications = set()
        all_related_topics = set()
        
        for idx in indices:
            kp = knowledge_points[idx]
            all_sources.add(kp.get('source', 'Unknown'))
            
            # Merge key concepts
            for concept in kp.get('key_concepts', []):
                all_key_concepts.add(concept)
            
            # Merge applications
            for app in kp.get('common_applications', []):
                all_applications.add(app)
            
            # Merge related topics
            for topic in kp.get('related_topics', []):
                all_related_topics.add(topic)
            
            # Use longer description if available
            if len(kp.get('description', '')) > len(merged.get('description', '')):
                merged['description'] = kp['description']
        
        # Update merged entry
        merged['source'] = ', '.join(sorted(all_sources))
        merged['key_concepts'] = sorted(list(all_key_concepts))
        merged['common_applications'] = sorted(list(all_applications))
        merged['related_topics'] = sorted(list(all_related_topics))
        
        return merged
    
    def resolve_id_conflicts(self, knowledge_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve ID conflicts by ensuring unique IDs.
        
        Args:
            knowledge_points: List of knowledge point dictionaries
        
        Returns:
            List with resolved IDs
        """
        used_ids: Set[str] = set()
        id_counter: Dict[str, int] = defaultdict(int)
        
        for kp in knowledge_points:
            original_id = kp.get('id', '')
            
            if original_id in used_ids:
                # Generate new ID
                base_id = original_id.rsplit('_', 1)[0] if '_' in original_id else original_id
                id_counter[base_id] += 1
                new_id = f"{base_id}_{id_counter[base_id]}"
                kp['id'] = new_id
                used_ids.add(new_id)
            else:
                used_ids.add(original_id)
        
        return knowledge_points
    
    def merge_all_sources(self) -> List[Dict[str, Any]]:
        """
        Merge all source files into a unified knowledge base.
        
        Returns:
            List of merged knowledge point dictionaries
        """
        print("Loading source files...")
        sources = self.load_source_files()
        
        if not sources:
            print("No source files found!")
            return []
        
        # Collect all knowledge points
        all_knowledge_points = []
        for source_data in sources:
            all_knowledge_points.extend(source_data.get('knowledge_points', []))
        
        print(f"\nTotal knowledge points before merging: {len(all_knowledge_points)}")
        
        # Detect duplicates
        print("\nDetecting duplicates...")
        duplicates = self.detect_duplicates(all_knowledge_points)
        print(f"Found {len(duplicates)} duplicate topics")
        
        # Process duplicates
        processed_indices: Set[int] = set()
        merged_points = []
        
        for topic, indices in duplicates.items():
            if not any(idx in processed_indices for idx in indices):
                merged_kp = self.merge_duplicate(all_knowledge_points, indices)
                merged_points.append(merged_kp)
                processed_indices.update(indices)
        
        # Add non-duplicate points
        for idx, kp in enumerate(all_knowledge_points):
            if idx not in processed_indices:
                merged_points.append(kp)
        
        print(f"Knowledge points after merging: {len(merged_points)}")
        
        # Resolve ID conflicts
        print("\nResolving ID conflicts...")
        merged_points = self.resolve_id_conflicts(merged_points)
        
        # Sort by category and topic
        merged_points.sort(key=lambda x: (
            x.get('category', ''),
            x.get('subcategory', ''),
            x.get('topic', '')
        ))
        
        return merged_points
    
    def save_merged_knowledge_base(self, knowledge_points: List[Dict[str, Any]]):
        """
        Save merged knowledge base to JSON file.
        
        Args:
            knowledge_points: List of merged knowledge point dictionaries
        """
        output_data = {
            "knowledge_points": knowledge_points,
            "total_points": len(knowledge_points),
            "merged_date": json.dumps({}, default=str),  # Will be set by json.dump
            "sources": list(set(kp.get('source', 'Unknown') for kp in knowledge_points))
        }
        
        # Create backup of existing file if it exists
        if self.output_path.exists():
            backup_path = self.output_path.with_suffix('.json.backup')
            import shutil
            shutil.copy(self.output_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(knowledge_points)} knowledge points to {self.output_path}")
        
        # Print statistics
        self.print_statistics(knowledge_points)
    
    def print_statistics(self, knowledge_points: List[Dict[str, Any]]):
        """Print statistics about the merged knowledge base."""
        categories = defaultdict(int)
        subcategories = defaultdict(int)
        sources = defaultdict(int)
        
        for kp in knowledge_points:
            categories[kp.get('category', 'Unknown')] += 1
            subcategories[kp.get('subcategory', 'Unknown')] += 1
            source = kp.get('source', 'Unknown')
            # Handle multiple sources
            for s in source.split(', '):
                sources[s.strip()] += 1
        
        print("\n=== Merged Knowledge Base Statistics ===")
        print(f"Total Knowledge Points: {len(knowledge_points)}")
        print(f"\nBy Category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print(f"\nBy Source:")
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count}")
        print(f"\nTotal Subcategories: {len(subcategories)}")


def main():
    """Main function to run the merger."""
    merger = KnowledgeBaseMerger()
    merged_points = merger.merge_all_sources()
    
    if merged_points:
        merger.save_merged_knowledge_base(merged_points)
    else:
        print("No knowledge points to merge!")


if __name__ == "__main__":
    main()
