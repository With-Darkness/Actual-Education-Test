"""Command-line interface for SAT Knowledge Matching System."""
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base import KnowledgeBase
from src.embeddings import EmbeddingGenerator
from src.retrieval import RAGRetriever
from src.reranker import BERTReranker
from src.query_enrichment import QueryEnricher
from src.evaluation import RAGEvaluator, SimpleEvaluator
from src.utils import format_search_results, load_demo_cases


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAT Knowledge Matching System - Find relevant SAT knowledge points"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Student question or topic to search"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5). Used when reranking is disabled."
    )
    
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=True,
        help="Enrich query with synonyms and expansions before retrieval (default: True)"
    )
    
    parser.add_argument(
        "--no-enrich",
        dest="enrich",
        action="store_false",
        help="Disable query enrichment"
    )
    
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use two-stage retrieval with BERT reranking"
    )
    
    parser.add_argument(
        "-m", "--candidates",
        type=int,
        default=20,
        help="Number of candidates to retrieve before reranking (default: 20). Only used with --rerank."
    )
    
    parser.add_argument(
        "-n", "--results",
        type=int,
        default=5,
        help="Number of final results after reranking (default: 5). Only used with --rerank."
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["markdown", "text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--kb-path",
        type=str,
        default=None,
        help="Path to knowledge base JSON file"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo cases from examples/demo_cases.json"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on demo cases using DeepEval"
    )
    
    parser.add_argument(
        "--eval-output",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results (default: evaluation_results.json)"
    )
    
    parser.add_argument(
        "--simple-eval",
        action="store_true",
        help="Use simple evaluator (no DeepEval required)"
    )
    
    args = parser.parse_args()
    
    # Determine knowledge base path
    if args.kb_path:
        kb_path = Path(args.kb_path)
    else:
        kb_path = Path(__file__).parent.parent / "data" / "sat_knowledge_base.json"
    
    if not kb_path.exists():
        print(f"Error: Knowledge base not found at {kb_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize system
    print("Initializing system...", file=sys.stderr)
    try:
        kb = KnowledgeBase(str(kb_path))
        embedder = EmbeddingGenerator()
        
        # Initialize reranker if requested
        reranker = None
        if args.rerank:
            try:
                print("Loading reranker model...", file=sys.stderr)
                reranker = BERTReranker()
            except Exception as e:
                print(f"Warning: Could not initialize reranker: {e}", file=sys.stderr)
                print("Continuing without reranking...", file=sys.stderr)
                args.rerank = False
        
        # Initialize query enricher if requested
        query_enricher = None
        if args.enrich:
            try:
                query_enricher = QueryEnricher(
                    enable_expansion=True,
                    enable_rewriting=True,
                    enable_llm_enhancement=False
                )
                print("Query enrichment enabled", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not initialize query enricher: {e}", file=sys.stderr)
                print("Continuing without query enrichment...", file=sys.stderr)
        
        retriever = RAGRetriever(kb, embedder, reranker=reranker, query_enricher=query_enricher)
        print("System ready!", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing system: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle stats flag
    if args.stats:
        stats = kb.get_statistics()
        index_stats = retriever.get_index_stats()
        
        print("\n=== Knowledge Base Statistics ===")
        print(f"Total Knowledge Points: {stats['total_points']}")
        print(f"\nCategories:")
        for cat, count in stats['categories'].items():
            print(f"  - {cat}: {count} points")
        print(f"\nIndex Statistics:")
        print(f"  - Vectors: {index_stats.get('num_vectors', 'N/A')}")
        print(f"  - Dimension: {index_stats.get('dimension', 'N/A')}")
        print(f"  - Similarity Algorithm: {index_stats.get('similarity_algorithm', 'N/A')}")
        print(f"  - Reranker Enabled: {index_stats.get('reranker_enabled', False)}")
        if index_stats.get('reranker_enabled') and 'reranker_model' in index_stats:
            reranker_info = index_stats['reranker_model']
            print(f"  - Reranker Model: {reranker_info.get('model_name', 'N/A')}")
        if 'query_enrichment' in index_stats:
            enrichment_info = index_stats['query_enrichment']
            print(f"  - Query Enrichment: Enabled")
            print(f"    - Expansion: {enrichment_info.get('expansion_enabled', False)}")
            print(f"    - Rewriting: {enrichment_info.get('rewriting_enabled', False)}")
        return
    
    # Handle demo flag
    if args.demo:
        demo_path = Path(__file__).parent.parent / "examples" / "demo_cases.json"
        if not demo_path.exists():
            print(f"Error: Demo cases file not found at {demo_path}", file=sys.stderr)
            sys.exit(1)
        
        demo_cases = load_demo_cases(str(demo_path))
        if not demo_cases:
            print("No demo cases found.", file=sys.stderr)
            return
        
        print(f"\n=== Running {len(demo_cases)} Demo Cases ===\n", file=sys.stderr)
        
        for i, case in enumerate(demo_cases, 1):
            query = case.get('query', '')
            expected_topics = case.get('expected_topics', [])
            
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Demo Case {i}: {query}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            
            # Use reranking if enabled
            if args.rerank and reranker:
                results = retriever.retrieve_with_reranking(
                    query, 
                    m=args.candidates, 
                    n=args.results,
                    enrich_query=args.enrich
                )
            else:
                results = retriever.retrieve(query, args.top_k, enrich_query=args.enrich)
            print(format_search_results(results, format_type=args.format))
            
            if expected_topics:
                print(f"\nExpected topics: {', '.join(expected_topics)}", file=sys.stderr)
                found_topics = [kp['topic'] for kp, _ in results]
                matches = [t for t in expected_topics if any(t.lower() in ft.lower() for ft in found_topics)]
                print(f"Matches found: {len(matches)}/{len(expected_topics)}", file=sys.stderr)
        
        return
    
    # Handle evaluation flag
    if args.evaluate:
        demo_path = Path(__file__).parent.parent / "examples" / "demo_cases.json"
        if not demo_path.exists():
            print(f"Error: Demo cases file not found at {demo_path}", file=sys.stderr)
            sys.exit(1)
        
        demo_cases = load_demo_cases(str(demo_path))
        if not demo_cases:
            print("No demo cases found.", file=sys.stderr)
            return
        
        print(f"\n=== Running Evaluation on {len(demo_cases)} Test Cases ===\n", file=sys.stderr)
        
        # Define retrieval function
        def retrieval_fn(query: str):
            if args.rerank and reranker:
                return retriever.retrieve_with_reranking(
                    query,
                    m=args.candidates,
                    n=args.results,
                    enrich_query=args.enrich
                )
            else:
                return retriever.retrieve(query, args.top_k, enrich_query=args.enrich)
        
        # Run evaluation
        if args.simple_eval:
            print("Using simple evaluator (no DeepEval)...", file=sys.stderr)
            evaluator = SimpleEvaluator()
            results = []
            for case in demo_cases:
                query = case.get('query', '')
                expected_topics = case.get('expected_topics', [])
                retrieved = retrieval_fn(query)
                eval_result = evaluator.evaluate_topic_match(retrieved, expected_topics)
                results.append({
                    "query": query,
                    "expected_topics": expected_topics,
                    "evaluation": eval_result
                })
            
            summary = {
                "total_cases": len(demo_cases),
                "average_precision": sum(r["evaluation"]["precision"] for r in results) / len(results),
                "average_recall": sum(r["evaluation"]["recall"] for r in results) / len(results),
                "average_f1": sum(r["evaluation"]["f1"] for r in results) / len(results),
                "results": results
            }
        else:
            try:
                evaluator = RAGEvaluator()
                batch_results = evaluator.evaluate_batch(demo_cases, retrieval_fn)
                summary = batch_results["summary"]
                results = batch_results["results"]
                
                # Print summary
                print("\n=== Evaluation Summary ===", file=sys.stderr)
                print(f"Total Cases: {summary['total_cases']}", file=sys.stderr)
                if "average_metrics" in summary:
                    for metric_name, metric_data in summary["average_metrics"].items():
                        print(f"\n{metric_name}:", file=sys.stderr)
                        print(f"  Mean: {metric_data['mean']:.3f}", file=sys.stderr)
                        print(f"  Min: {metric_data['min']:.3f}", file=sys.stderr)
                        print(f"  Max: {metric_data['max']:.3f}", file=sys.stderr)
                        print(f"  Count: {metric_data['count']}", file=sys.stderr)
                
                evaluator.save_evaluation_results(
                    {"summary": summary, "results": results},
                    args.eval_output
                )
            except ImportError:
                print("Error: DeepEval not installed. Install with: pip install deepeval", file=sys.stderr)
                print("Or use --simple-eval for basic evaluation", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error during evaluation: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        # Save results
        output_data = {
            "summary": summary,
            "results": results,
            "timestamp": datetime.now().isoformat() if 'datetime' in dir() else None
        }
        
        output_path = Path(args.eval_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation results saved to {output_path}", file=sys.stderr)
        return
    
    # Handle query
    if not args.query:
        parser.print_help()
        return
    
    # Perform search
    try:
        # Use reranking if enabled
        if args.rerank and reranker:
            enrich_status = "with query enrichment" if args.enrich else "without query enrichment"
            print(f"Using two-stage retrieval {enrich_status}: retrieving {args.candidates} candidates, reranking to {args.results} results...", file=sys.stderr)
            results = retriever.retrieve_with_reranking(
                args.query, 
                m=args.candidates, 
                n=args.results,
                enrich_query=args.enrich
            )
        else:
            enrich_status = "with query enrichment" if args.enrich else "without query enrichment"
            print(f"Retrieving {enrich_status}...", file=sys.stderr)
            results = retriever.retrieve(args.query, args.top_k, enrich_query=args.enrich)
        
        output = format_search_results(results, format_type=args.format)
        print(output)
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
