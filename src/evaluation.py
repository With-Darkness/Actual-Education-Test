"""
Evaluation Pipeline using DeepEval Framework

Evaluates RAG system retrieval quality using various metrics:
- Answer Relevance
- Contextual Precision
- Contextual Recall
- Faithfulness
- Custom SAT-specific metrics
"""
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime

try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("Warning: DeepEval not installed. Install with: pip install deepeval")


class RAGEvaluator:
    """Evaluator for RAG system using DeepEval framework."""
    
    def __init__(self):
        """Initialize evaluator."""
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is not installed. Install it with: pip install deepeval"
            )
        
        # Initialize metrics
        self.answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
        self.contextual_precision = ContextualPrecisionMetric(threshold=0.7)
        self.contextual_recall = ContextualRecallMetric(threshold=0.7)
        self.faithfulness = FaithfulnessMetric(threshold=0.7)
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_knowledge_points: List[Tuple[Dict[str, Any], float]],
                          expected_topics: Optional[List[str]] = None,
                          expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval results for a single query.
        
        Args:
            query: Student question
            retrieved_knowledge_points: List of (knowledge_point, score) tuples
            expected_topics: List of expected topic names (for topic-based evaluation)
            expected_answer: Expected answer text (for answer-based evaluation)
        
        Returns:
            Dictionary with evaluation metrics and scores
        """
        if not retrieved_knowledge_points:
            return {
                "query": query,
                "num_retrieved": 0,
                "metrics": {},
                "error": "No results retrieved"
            }
        
        # Prepare context from retrieved knowledge points
        context = self._format_context(retrieved_knowledge_points)
        
        # Generate answer from retrieved knowledge points
        answer = self._generate_answer(query, retrieved_knowledge_points)
        
        # Calculate metrics
        metrics = {}
        
        # Topic-based evaluation (if expected topics provided)
        if expected_topics:
            topic_metrics = self._evaluate_topic_match(
                retrieved_knowledge_points, 
                expected_topics
            )
            metrics.update(topic_metrics)
        
        # Answer relevancy (if expected answer provided)
        if expected_answer:
            try:
                test_case = LLMTestCase(
                    input=query,
                    actual_output=answer,
                    expected_output=expected_answer,
                    context=context
                )
                relevancy_score = self.answer_relevancy.measure(test_case)
                metrics["answer_relevancy"] = {
                    "score": relevancy_score,
                    "passed": self.answer_relevancy.success
                }
            except Exception as e:
                metrics["answer_relevancy"] = {"error": str(e)}
        
        # Contextual precision and recall
        try:
            precision_test = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_answer or query,
                context=context
            )
            precision_score = self.contextual_precision.measure(precision_test)
            metrics["contextual_precision"] = {
                "score": precision_score,
                "passed": self.contextual_precision.success
            }
        except Exception as e:
            metrics["contextual_precision"] = {"error": str(e)}
        
        try:
            recall_test = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_answer or query,
                context=context
            )
            recall_score = self.contextual_recall.measure(recall_test)
            metrics["contextual_recall"] = {
                "score": recall_score,
                "passed": self.contextual_recall.success
            }
        except Exception as e:
            metrics["contextual_recall"] = {"error": str(e)}
        
        # Faithfulness (if answer provided)
        if answer:
            try:
                faithfulness_test = LLMTestCase(
                    input=query,
                    actual_output=answer,
                    expected_output=expected_answer or query,
                    context=context
                )
                faithfulness_score = self.faithfulness.measure(faithfulness_test)
                metrics["faithfulness"] = {
                    "score": faithfulness_score,
                    "passed": self.faithfulness.success
                }
            except Exception as e:
                metrics["faithfulness"] = {"error": str(e)}
        
        # Custom metrics
        custom_metrics = self._calculate_custom_metrics(
            query, 
            retrieved_knowledge_points,
            expected_topics
        )
        metrics.update(custom_metrics)
        
        return {
            "query": query,
            "num_retrieved": len(retrieved_knowledge_points),
            "answer": answer,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate_batch(self, 
                      test_cases: List[Dict[str, Any]],
                      retrieval_function) -> Dict[str, Any]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries with:
                - query: Student question
                - expected_topics: List of expected topics (optional)
                - expected_answer: Expected answer (optional)
            retrieval_function: Function that takes query and returns retrieval results
        
        Returns:
            Dictionary with batch evaluation results
        """
        results = []
        total_metrics = {
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
            "faithfulness": [],
            "topic_precision": [],
            "topic_recall": [],
            "average_similarity": []
        }
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case.get('query', '')
            expected_topics = test_case.get('expected_topics', [])
            expected_answer = test_case.get('expected_answer', '')
            
            print(f"Evaluating test case {i}/{len(test_cases)}: {query[:50]}...")
            
            # Get retrieval results
            retrieved = retrieval_function(query)
            
            # Evaluate
            evaluation = self.evaluate_retrieval(
                query=query,
                retrieved_knowledge_points=retrieved,
                expected_topics=expected_topics,
                expected_answer=expected_answer
            )
            
            results.append(evaluation)
            
            # Aggregate metrics
            metrics = evaluation.get('metrics', {})
            for metric_name in total_metrics.keys():
                if metric_name in metrics:
                    score = metrics[metric_name].get('score', 0)
                    if isinstance(score, (int, float)):
                        total_metrics[metric_name].append(score)
        
        # Calculate averages
        summary = {
            "total_cases": len(test_cases),
            "average_metrics": {}
        }
        
        for metric_name, scores in total_metrics.items():
            if scores:
                summary["average_metrics"][metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        return {
            "summary": summary,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_context(self, knowledge_points: List[Tuple[Dict[str, Any], float]]) -> str:
        """Format retrieved knowledge points as context string."""
        context_parts = []
        for kp, score in knowledge_points:
            context_parts.append(
                f"Topic: {kp.get('topic', '')}\n"
                f"Description: {kp.get('description', '')}\n"
                f"Key Concepts: {', '.join(kp.get('key_concepts', [])[:3])}\n"
                f"Relevance: {score:.2%}\n"
            )
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, query: str, knowledge_points: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate answer from retrieved knowledge points."""
        if not knowledge_points:
            return "I couldn't find relevant information to answer your question."
        
        # Use top result to generate answer
        top_kp, top_score = knowledge_points[0]
        
        answer_parts = [
            f"Based on the SAT curriculum, here's information about {top_kp.get('topic', 'this topic')}:",
            top_kp.get('description', ''),
        ]
        
        # Add key concepts
        key_concepts = top_kp.get('key_concepts', [])[:3]
        if key_concepts:
            answer_parts.append(f"\nKey concepts: {', '.join(key_concepts)}")
        
        # Add example if available
        example = top_kp.get('example_problem', '')
        if example:
            answer_parts.append(f"\nExample: {example}")
        
        return "\n".join(answer_parts)
    
    def _evaluate_topic_match(self, 
                             retrieved_knowledge_points: List[Tuple[Dict[str, Any], float]],
                             expected_topics: List[str]) -> Dict[str, Any]:
        """Evaluate topic-based matching."""
        retrieved_topics = [kp.get('topic', '') for kp, _ in retrieved_knowledge_points]
        
        # Normalize topics for comparison
        retrieved_normalized = [t.lower().strip() for t in retrieved_topics]
        expected_normalized = [t.lower().strip() for t in expected_topics]
        
        # Calculate precision and recall
        matches = sum(1 for exp_topic in expected_normalized 
                     if any(exp_topic in ret_topic or ret_topic in exp_topic 
                           for ret_topic in retrieved_normalized))
        
        precision = matches / len(retrieved_topics) if retrieved_topics else 0.0
        recall = matches / len(expected_topics) if expected_topics else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "topic_precision": {
                "score": precision,
                "passed": precision >= 0.5
            },
            "topic_recall": {
                "score": recall,
                "passed": recall >= 0.5
            },
            "topic_f1": {
                "score": f1,
                "passed": f1 >= 0.5
            },
            "matched_topics": matches,
            "expected_topics": expected_topics,
            "retrieved_topics": retrieved_topics[:len(expected_topics)]
        }
    
    def _calculate_custom_metrics(self,
                                 query: str,
                                 retrieved_knowledge_points: List[Tuple[Dict[str, Any], float]],
                                 expected_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate custom SAT-specific metrics."""
        metrics = {}
        
        # Average similarity score
        if retrieved_knowledge_points:
            avg_similarity = sum(score for _, score in retrieved_knowledge_points) / len(retrieved_knowledge_points)
            metrics["average_similarity"] = {
                "score": avg_similarity,
                "passed": avg_similarity >= 0.5
            }
        
        # Top result relevance
        if retrieved_knowledge_points:
            top_score = retrieved_knowledge_points[0][1]
            metrics["top_result_relevance"] = {
                "score": top_score,
                "passed": top_score >= 0.7
            }
        
        # Diversity (how many different categories)
        categories = set(kp.get('category', '') for kp, _ in retrieved_knowledge_points)
        metrics["category_diversity"] = {
            "score": len(categories),
            "max_possible": 4  # Math, Reading, Writing, Strategy
        }
        
        return metrics
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to {output_file}")


class SimpleEvaluator:
    """
    Simple evaluator that doesn't require DeepEval.
    Useful for basic evaluation without LLM dependencies.
    """
    
    @staticmethod
    def evaluate_topic_match(retrieved_knowledge_points: List[Tuple[Dict[str, Any], float]],
                            expected_topics: List[str]) -> Dict[str, Any]:
        """Simple topic-based evaluation."""
        retrieved_topics = [kp.get('topic', '') for kp, _ in retrieved_knowledge_points]
        
        retrieved_normalized = [t.lower().strip() for t in retrieved_topics]
        expected_normalized = [t.lower().strip() for t in expected_topics]
        
        matches = sum(1 for exp_topic in expected_normalized 
                     if any(exp_topic in ret_topic or ret_topic in exp_topic 
                           for ret_topic in retrieved_normalized))
        
        precision = matches / len(retrieved_topics) if retrieved_topics else 0.0
        recall = matches / len(expected_topics) if expected_topics else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches,
            "expected_count": len(expected_topics),
            "retrieved_count": len(retrieved_topics)
        }
