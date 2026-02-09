"""
Main script to run all data scrapers and merge results.

This script:
1. Runs Khan Academy scraper
2. Runs College Board scraper
3. Runs Educational Content scraper
4. Merges all results into a unified knowledge base
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.khan_academy_scraper import KhanAcademyScraper
from src.data_processing.college_board_scraper import CollegeBoardScraper
from src.data_processing.educational_content_scraper import EducationalContentScraper
from src.data_processing.merge_sources import KnowledgeBaseMerger


def main():
    """Run all scrapers and merge results."""
    print("=" * 60)
    print("SAT Knowledge Base Expansion Script")
    print("=" * 60)
    print()
    
    output_dir = "data/processed"
    
    # Step 1: Run Khan Academy scraper
    print("\n[1/4] Running Khan Academy scraper...")
    print("-" * 60)
    try:
        ka_scraper = KhanAcademyScraper(output_dir=output_dir)
        ka_points = ka_scraper.scrape_all_topics()
        ka_scraper.save_to_json(ka_points)
        print(f"✓ Successfully extracted {len(ka_points)} points from Khan Academy")
    except Exception as e:
        print(f"✗ Error in Khan Academy scraper: {e}")
        ka_points = []
    
    # Step 2: Run College Board scraper
    print("\n[2/4] Running College Board scraper...")
    print("-" * 60)
    try:
        cb_scraper = CollegeBoardScraper(output_dir=output_dir)
        cb_points = cb_scraper.scrape_all_topics()
        cb_scraper.save_to_json(cb_points)
        print(f"✓ Successfully extracted {len(cb_points)} points from College Board")
    except Exception as e:
        print(f"✗ Error in College Board scraper: {e}")
        cb_points = []
    
    # Step 3: Run Educational Content scraper
    print("\n[3/4] Running Educational Content scraper...")
    print("-" * 60)
    try:
        edu_scraper = EducationalContentScraper(output_dir=output_dir)
        edu_points = edu_scraper.scrape_all_topics()
        edu_scraper.save_to_json(edu_points)
        print(f"✓ Successfully extracted {len(edu_points)} points from Educational Resources")
    except Exception as e:
        print(f"✗ Error in Educational Content scraper: {e}")
        edu_points = []
    
    # Step 4: Merge all sources
    print("\n[4/4] Merging all sources...")
    print("-" * 60)
    try:
        merger = KnowledgeBaseMerger(
            processed_dir=output_dir,
            output_path="data/sat_knowledge_base.json"
        )
        merged_points = merger.merge_all_sources()
        
        if merged_points:
            merger.save_merged_knowledge_base(merged_points)
            print("\n" + "=" * 60)
            print("✓ Knowledge base expansion completed successfully!")
            print("=" * 60)
            print(f"\nExpanded knowledge base saved to: data/sat_knowledge_base_expanded.json")
            print(f"Total knowledge points: {len(merged_points)}")
        else:
            print("✗ No knowledge points to merge!")
    
    except Exception as e:
        print(f"✗ Error merging sources: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
