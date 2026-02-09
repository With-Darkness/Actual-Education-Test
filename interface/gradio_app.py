"""Gradio web interface for SAT Knowledge Matching System."""
import gradio as gr
import sys
import os
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


# Initialize components
def initialize_system(use_reranker: bool = True, use_enrichment: bool = True):
    """
    Initialize the RAG system components.
    
    Args:
        use_reranker: Whether to initialize BERT reranker for two-stage retrieval
        use_enrichment: Whether to initialize query enricher
    """
    kb_path = Path(__file__).parent.parent / "data" / "sat_knowledge_base.json"
    
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")
    
    kb = KnowledgeBase(str(kb_path))
    embedder = EmbeddingGenerator()
    
    # Initialize reranker if requested
    reranker = None
    if use_reranker:
        try:
            reranker = BERTReranker()
        except Exception as e:
            print(f"Warning: Could not initialize reranker: {e}")
            print("Continuing without reranker...")
    
    # Initialize query enricher if requested
    query_enricher = None
    if use_enrichment:
        try:
            query_enricher = QueryEnricher(
                enable_expansion=True,
                enable_rewriting=True,
                enable_llm_enhancement=False
            )
            print("Query enrichment enabled")
        except Exception as e:
            print(f"Warning: Could not initialize query enricher: {e}")
            print("Continuing without query enrichment...")
    
    retriever = RAGRetriever(kb, embedder, reranker=reranker, query_enricher=query_enricher)
    
    return kb, embedder, retriever, reranker, query_enricher


# Initialize system (will load model on first import)
print("Initializing SAT Knowledge Matching System...")
try:
    KB, EMBEDDER, RETRIEVER, RERANKER, QUERY_ENRICHER = initialize_system(
        use_reranker=True, 
        use_enrichment=True
    )
    print("System initialized successfully!")
except Exception as e:
    print(f"Error initializing system: {e}")
    KB, EMBEDDER, RETRIEVER, RERANKER, QUERY_ENRICHER = None, None, None, None, None


def search_knowledge(query: str, use_reranking: bool = True, m_candidates: int = 20, 
                    n_results: int = 5, enrich_query: bool = True, show_details: bool = True):
    """
    Search for relevant SAT knowledge points with optional reranking and query enrichment.
    
    Args:
        query: Student question or topic
        use_reranking: Whether to use two-stage retrieval with reranking
        m_candidates: Number of candidates to retrieve before reranking (if reranking enabled)
        n_results: Number of final results to return
        enrich_query: Whether to enrich the query before retrieval
        show_details: Whether to show detailed information
    """
    if not query or not query.strip():
        return "Please enter a question or topic to search."
    
    if RETRIEVER is None:
        return "Error: System not initialized. Please check the logs."
    
    try:
        # Use two-stage retrieval with reranking if enabled and reranker available
        if use_reranking and RERANKER is not None:
            results = RETRIEVER.retrieve_with_reranking(
                query.strip(), 
                m=m_candidates, 
                n=n_results,
                enrich_query=enrich_query
            )
        else:
            # Fallback to regular retrieval
            results = RETRIEVER.retrieve(query.strip(), top_k=n_results, enrich_query=enrich_query)
        
        if not results:
            return "No relevant knowledge points found. Try rephrasing your question."
        
        if show_details:
            return format_search_results(results, format_type="markdown")
        else:
            # Simplified output
            output = []
            for i, (kp, similarity) in enumerate(results, 1):
                output.append(
                    f"**{i}. {kp['topic']}** ({similarity:.1%} match)\n"
                    f"   {kp['category']} > {kp['subcategory']}\n"
                )
            return "\n".join(output)
    
    except Exception as e:
        return f"Error during search: {str(e)}"


def get_system_info():
    """Get information about the knowledge base."""
    if KB is None:
        return "System not initialized."
    
    stats = KB.get_statistics()
    index_stats = RETRIEVER.get_index_stats() if RETRIEVER else {}
    
    info = f"""
# System Information

## Knowledge Base Statistics
- **Total Knowledge Points**: {stats['total_points']}
- **Categories**: {', '.join(stats['categories'].keys())}

## Category Breakdown
"""
    for cat, count in stats['categories'].items():
        info += f"- **{cat}**: {count} points\n"
    
    if index_stats:
        info += f"""
## Retrieval System
- **Similarity Algorithm**: {index_stats.get('similarity_algorithm', 'N/A')}
- **Vectors Indexed**: {index_stats.get('num_vectors', 'N/A')}
- **Embedding Dimension**: {index_stats.get('dimension', 'N/A')}
- **Reranker Enabled**: {'Yes' if index_stats.get('reranker_enabled', False) else 'No'}
"""
        
        if index_stats.get('reranker_enabled') and 'reranker_model' in index_stats:
            reranker_info = index_stats['reranker_model']
            info += f"""
## Reranker Model
- **Model**: {reranker_info.get('model_name', 'N/A')}
- **Type**: {reranker_info.get('model_type', 'N/A')}
- **Max Length**: {reranker_info.get('max_length', 'N/A')}
"""
        
        if 'query_enrichment' in index_stats:
            enrichment_info = index_stats['query_enrichment']
            info += f"""
## Query Enrichment
- **Expansion Enabled**: {enrichment_info.get('expansion_enabled', False)}
- **Rewriting Enabled**: {enrichment_info.get('rewriting_enabled', False)}
- **Synonyms Available**: {enrichment_info.get('synonyms_count', 0)}
- **Topic Expansions**: {enrichment_info.get('topic_expansions_count', 0)}
"""
    
    return info


def run_evaluation(use_reranking: bool, m_candidates: int, n_results: int, 
                  enrich_query: bool, use_simple_eval: bool):
    """Run evaluation on demo cases."""
    if RETRIEVER is None:
        return "Error: System not initialized. Please check the logs."
    
    try:
        demo_path = Path(__file__).parent.parent / "examples" / "demo_cases.json"
        if not demo_path.exists():
            return f"Error: Demo cases file not found at {demo_path}"
        
        demo_cases = load_demo_cases(str(demo_path))
        if not demo_cases:
            return "No demo cases found."
        
        # Define retrieval function
        def retrieval_fn(query: str):
            if use_reranking and RERANKER is not None:
                return RETRIEVER.retrieve_with_reranking(
                    query,
                    m=m_candidates,
                    n=n_results,
                    enrich_query=enrich_query
                )
            else:
                return RETRIEVER.retrieve(query, top_k=n_results, enrich_query=enrich_query)
        
        # Run evaluation
        if use_simple_eval:
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
            
            avg_precision = sum(r["evaluation"]["precision"] for r in results) / len(results)
            avg_recall = sum(r["evaluation"]["recall"] for r in results) / len(results)
            avg_f1 = sum(r["evaluation"]["f1"] for r in results) / len(results)
            
            output = f"""
# Evaluation Results (Simple Evaluator)

## Summary
- **Total Cases**: {len(demo_cases)}
- **Average Precision**: {avg_precision:.3f}
- **Average Recall**: {avg_recall:.3f}
- **Average F1 Score**: {avg_f1:.3f}

## Individual Results
"""
            for i, result in enumerate(results, 1):
                eval_data = result["evaluation"]
                output += f"""
### Case {i}: {result['query']}
- **Expected Topics**: {', '.join(result['expected_topics'])}
- **Precision**: {eval_data['precision']:.3f}
- **Recall**: {eval_data['recall']:.3f}
- **F1**: {eval_data['f1']:.3f}
- **Matches**: {eval_data['matches']}/{eval_data['expected_count']}
"""
        else:
            try:
                evaluator = RAGEvaluator()
                batch_results = evaluator.evaluate_batch(demo_cases, retrieval_fn)
                summary = batch_results["summary"]
                
                output = f"""
# Evaluation Results (DeepEval)

## Summary
- **Total Cases**: {summary['total_cases']}
"""
                if "average_metrics" in summary:
                    output += "\n### Average Metrics\n"
                    for metric_name, metric_data in summary["average_metrics"].items():
                        output += f"""
- **{metric_name.replace('_', ' ').title()}**:
  - Mean: {metric_data['mean']:.3f}
  - Min: {metric_data['min']:.3f}
  - Max: {metric_data['max']:.3f}
"""
                
                output += "\n### Individual Results\n"
                for i, result in enumerate(batch_results["results"], 1):
                    output += f"\n**Case {i}**: {result['query']}\n"
                    metrics = result.get('metrics', {})
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and 'score' in metric_data:
                            score = metric_data['score']
                            passed = metric_data.get('passed', False)
                            output += f"- {metric_name}: {score:.3f} {'✅' if passed else '❌'}\n"
            
            except ImportError:
                return "Error: DeepEval not installed. Install with: pip install deepeval\nOr use Simple Evaluator."
            except Exception as e:
                return f"Error during evaluation: {str(e)}"
        
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface with tabs
with gr.Blocks(title="SAT Knowledge Matching System", theme=gr.themes.Soft()) as iface:
    gr.Markdown("""
    # 🎓 SAT Knowledge Matching System
    
    Enter a student question or topic to find relevant SAT knowledge points and concepts.
    The system uses semantic search to match your query with curriculum content.
    """)
    
    with gr.Tabs():
        # Search Tab
        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Student Question or Topic",
                        placeholder="e.g., 'How do I solve quadratic equations?' or 'subject-verb agreement'",
                        lines=3
                    )
                    
                    with gr.Accordion("Retrieval Settings", open=False):
                        enrich_query_checkbox = gr.Checkbox(
                            value=True,
                            label="Enrich Query",
                            info="Expand query with synonyms and related terms for better retrieval"
                        )
                        
                        use_reranking_checkbox = gr.Checkbox(
                            value=True,
                            label="Use Reranking (Two-Stage Retrieval)",
                            info="Enable BERT reranker for more accurate results"
                        )
                        
                        with gr.Row():
                            m_candidates_slider = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                step=5,
                                label="Candidates (m)",
                                info="Number of candidates retrieved before reranking"
                            )
                            n_results_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Final Results (n)",
                                info="Number of results returned after reranking"
                            )
                    
                    with gr.Row():
                        show_details_checkbox = gr.Checkbox(
                            value=True,
                            label="Show Detailed Information"
                        )
                    
                    search_btn = gr.Button("Search", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### System Information")
                    info_btn = gr.Button("Show System Info")
                    info_output = gr.Markdown()
            
            output = gr.Markdown(label="Relevant Knowledge Points")
            
            # Examples
            gr.Markdown("### Example Queries")
            examples = gr.Examples(
                examples=[
                    ["How do I solve quadratic equations?"],
                    ["What is subject-verb agreement?"],
                    ["Explain the Pythagorean theorem"],
                    ["How to find the main idea of a passage?"],
                    ["What are context clues?"]
                ],
                inputs=query_input
            )
            
            # Event handlers
            search_btn.click(
                fn=search_knowledge,
                inputs=[query_input, use_reranking_checkbox, m_candidates_slider, 
                       n_results_slider, enrich_query_checkbox, show_details_checkbox],
                outputs=output
            )
            
            query_input.submit(
                fn=search_knowledge,
                inputs=[query_input, use_reranking_checkbox, m_candidates_slider, 
                       n_results_slider, enrich_query_checkbox, show_details_checkbox],
                outputs=output
            )
            
            info_btn.click(
                fn=get_system_info,
                outputs=info_output
            )
        
        # Evaluation Tab
        with gr.Tab("Evaluation"):
            gr.Markdown("""
            ## System Evaluation
            
            Evaluate the RAG system's retrieval quality using DeepEval framework.
            Tests against demo cases with expected topics.
            """)
            
            with gr.Row():
                with gr.Column():
                    eval_use_reranking = gr.Checkbox(
                        value=True,
                        label="Use Reranking for Evaluation"
                    )
                    eval_enrich_query = gr.Checkbox(
                        value=True,
                        label="Use Query Enrichment for Evaluation"
                    )
                    eval_use_simple = gr.Checkbox(
                        value=False,
                        label="Use Simple Evaluator (No DeepEval)",
                        info="Use basic topic matching instead of DeepEval metrics"
                    )
                    
                    with gr.Row():
                        eval_m_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Candidates (m)"
                        )
                        eval_n_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Results (n)"
                        )
                    
                    eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            
            eval_output = gr.Markdown(label="Evaluation Results")
            
            eval_btn.click(
                fn=run_evaluation,
                inputs=[eval_use_reranking, eval_m_slider, eval_n_slider, 
                       eval_enrich_query, eval_use_simple],
                outputs=eval_output
            )


if __name__ == "__main__":
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
