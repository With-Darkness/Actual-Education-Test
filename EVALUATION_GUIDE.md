# Evaluation Pipeline Guide

## Overview

The evaluation pipeline uses DeepEval framework to assess the quality of RAG system retrieval results. It provides comprehensive metrics for measuring retrieval accuracy, relevance, and faithfulness.

## DeepEval Framework

DeepEval is a framework for evaluating LLM outputs, particularly useful for RAG systems. It provides various metrics:

- **Answer Relevancy**: Measures how relevant the answer is to the query
- **Contextual Precision**: Measures precision of retrieved context
- **Contextual Recall**: Measures recall of relevant context
- **Faithfulness**: Measures if the answer is faithful to the context

## Implementation

### Evaluation Module (`src/evaluation.py`)

The evaluation module includes:

1. **RAGEvaluator**: Full DeepEval-based evaluator
   - Uses DeepEval metrics
   - Evaluates single queries and batches
   - Generates comprehensive reports

2. **SimpleEvaluator**: Lightweight evaluator
   - No DeepEval dependency
   - Topic-based matching
   - Precision, Recall, F1 scores

### Metrics Calculated

#### DeepEval Metrics (if available)
- Answer Relevancy Score
- Contextual Precision Score
- Contextual Recall Score
- Faithfulness Score

#### Custom Metrics
- Topic Precision: How many retrieved topics match expected topics
- Topic Recall: How many expected topics were retrieved
- Topic F1: Harmonic mean of precision and recall
- Average Similarity: Average similarity scores of retrieved results
- Top Result Relevance: Relevance score of top result
- Category Diversity: Number of different categories retrieved

## Usage

### CLI Evaluation

#### Full DeepEval Evaluation

```bash
# Run evaluation with DeepEval
python interface/cli.py --evaluate --rerank -m 20 -n 5

# With query enrichment
python interface/cli.py --evaluate --enrich --rerank

# Save results to custom file
python interface/cli.py --evaluate --eval-output my_results.json
```

#### Simple Evaluation (No DeepEval)

```bash
# Use simple evaluator (no DeepEval required)
python interface/cli.py --evaluate --simple-eval
```

### Gradio Interface

1. Open the web interface
2. Navigate to the "Evaluation" tab
3. Configure evaluation settings:
   - Use Reranking
   - Use Query Enrichment
   - Use Simple Evaluator (optional)
   - Set m and n values
4. Click "Run Evaluation"
5. View results

### Python API

```python
from src.evaluation import RAGEvaluator
from src.retrieval import RAGRetriever

# Initialize evaluator
evaluator = RAGEvaluator()

# Define retrieval function
def retrieval_fn(query: str):
    return retriever.retrieve(query, top_k=5)

# Evaluate single query
result = evaluator.evaluate_retrieval(
    query="How do I solve quadratic equations?",
    retrieved_knowledge_points=retrieval_fn("How do I solve quadratic equations?"),
    expected_topics=["Quadratic Equations", "Factoring"],
    expected_answer="Quadratic equations can be solved using..."
)

# Evaluate batch
test_cases = [
    {
        "query": "How do I solve quadratic equations?",
        "expected_topics": ["Quadratic Equations", "Factoring"],
        "expected_answer": "..."
    },
    # ... more test cases
]

batch_results = evaluator.evaluate_batch(test_cases, retrieval_fn)
```

## Evaluation Output

### Summary Metrics

```json
{
  "summary": {
    "total_cases": 5,
    "average_metrics": {
      "answer_relevancy": {
        "mean": 0.85,
        "min": 0.72,
        "max": 0.95,
        "count": 5
      },
      "topic_precision": {
        "mean": 0.80,
        "min": 0.60,
        "max": 1.0,
        "count": 5
      }
    }
  },
  "results": [...]
}
```

### Individual Case Results

Each test case includes:
- Query
- Retrieved knowledge points
- Generated answer
- All metric scores
- Pass/fail status for each metric

## Test Cases Format

Test cases are stored in `examples/demo_cases.json`:

```json
{
  "demo_cases": [
    {
      "id": "demo_001",
      "query": "How do I solve quadratic equations?",
      "expected_topics": ["Quadratic Equations", "Factoring"],
      "expected_answer": "Quadratic equations can be solved...",
      "category": "Math"
    }
  ]
}
```

## Metrics Interpretation

### Score Ranges

- **0.0 - 0.5**: Poor retrieval
- **0.5 - 0.7**: Acceptable retrieval
- **0.7 - 0.9**: Good retrieval
- **0.9 - 1.0**: Excellent retrieval

### Thresholds

Default thresholds (can be adjusted):
- Answer Relevancy: 0.7
- Contextual Precision: 0.7
- Contextual Recall: 0.7
- Faithfulness: 0.7
- Topic Precision: 0.5
- Topic Recall: 0.5

## Customization

### Adjusting Thresholds

Edit `src/evaluation.py`:

```python
self.answer_relevancy = AnswerRelevancyMetric(threshold=0.8)  # Higher threshold
```

### Adding Custom Metrics

Add new metrics to `_calculate_custom_metrics()`:

```python
def _calculate_custom_metrics(self, ...):
    metrics = {}
    # Your custom metric
    metrics["custom_metric"] = {
        "score": calculated_score,
        "passed": calculated_score >= threshold
    }
    return metrics
```

## Performance

- **Evaluation Time**: ~2-5 seconds per test case (with DeepEval)
- **Simple Evaluation**: ~0.1-0.5 seconds per test case
- **Batch Evaluation**: Scales linearly with number of cases

## Troubleshooting

### DeepEval Not Installed

If you get import errors:
```bash
pip install deepeval
```

Or use simple evaluator:
```bash
python interface/cli.py --evaluate --simple-eval
```

### API Key Required

Some DeepEval metrics may require API keys:
- Set environment variables as needed
- Or use simple evaluator for basic metrics

### Evaluation Errors

If evaluation fails:
1. Check test cases format
2. Verify retrieval function works
3. Check DeepEval installation
4. Review error messages in console

## Best Practices

1. **Use Expected Topics**: Always provide expected topics for better evaluation
2. **Batch Evaluation**: Evaluate multiple cases for statistical significance
3. **Compare Configurations**: Run evaluations with/without reranking and enrichment
4. **Regular Evaluation**: Run evaluations after knowledge base updates
5. **Track Metrics**: Save results and track improvements over time

## Example Workflow

```bash
# 1. Run evaluation with reranking
python interface/cli.py --evaluate --rerank -m 20 -n 5

# 2. Run evaluation without reranking (for comparison)
python interface/cli.py --evaluate -k 5

# 3. Compare results
# Check evaluation_results.json files

# 4. Use simple evaluator for quick checks
python interface/cli.py --evaluate --simple-eval
```

## Integration with CI/CD

You can integrate evaluation into automated testing:

```python
# In your test script
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate_batch(test_cases, retrieval_fn)

# Check if metrics meet thresholds
avg_precision = results["summary"]["average_metrics"]["topic_precision"]["mean"]
assert avg_precision >= 0.7, f"Precision too low: {avg_precision}"
```

## Future Enhancements

Potential improvements:
- [ ] Custom evaluation metrics for SAT-specific tasks
- [ ] A/B testing framework
- [ ] Evaluation dashboard
- [ ] Automated evaluation on knowledge base updates
- [ ] Integration with experiment tracking tools
