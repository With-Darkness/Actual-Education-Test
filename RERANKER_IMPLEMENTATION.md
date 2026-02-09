# BERT Reranker Implementation Guide

## Overview

A two-stage retrieval system has been implemented with BERT-based reranking for improved accuracy. The system uses:

1. **Stage 1 (FAISS)**: Fast semantic search using cosine similarity to retrieve `m` candidates
2. **Stage 2 (BERT Reranker)**: Accurate reranking using cross-encoder to select top `n` results

## Architecture

### Components

1. **BERTReranker** (`src/reranker.py`)
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` model
   - Cross-encoder architecture for accurate query-document scoring
   - Normalizes scores to 0-1 range
   - Combines rerank scores with initial FAISS scores (70% rerank, 30% initial)

2. **RAGRetriever** (Updated `src/retrieval.py`)
   - Added `retrieve_with_reranking()` method
   - Two-stage pipeline: FAISS → BERT Reranker
   - Backward compatible with original `retrieve()` method

3. **Gradio Interface** (Updated `interface/gradio_app.py`)
   - Added controls for `m` (candidates) and `n` (final results)
   - Toggle to enable/disable reranking
   - Shows reranker status in system info

4. **CLI** (Updated `interface/cli.py`)
   - Added `--rerank` flag
   - Added `-m/--candidates` and `-n/--results` parameters
   - Backward compatible with existing `-k` parameter

## Model Selection

### Chosen Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Why this model?**
- **Fast**: Smaller model (6 layers) for quick inference
- **Accurate**: Trained specifically on MS MARCO dataset for retrieval/reranking
- **Balanced**: Good trade-off between speed and quality
- **Proven**: Widely used in production RAG systems

### Alternative Models Available

The `BERTReranker` class supports alternative models:

- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Higher quality, slower
- `BAAI/bge-reranker-base`: General purpose reranker
- Custom models: Can be specified during initialization

## Similarity Search Algorithm

### Current Implementation: Cosine Similarity (L2 Normalized)

**Why Cosine Similarity?**
- **Semantic Accuracy**: Better captures semantic relationships than Euclidean distance
- **Normalization**: Handles varying text lengths well
- **Standard**: Industry standard for semantic search
- **Efficient**: Fast computation with FAISS

**Implementation Details:**
- Embeddings are L2-normalized
- FAISS uses L2 distance on normalized vectors (equivalent to cosine distance)
- Scores converted to similarity (0-1 scale)

### Why Not Other Algorithms?

- **Euclidean Distance**: Less accurate for semantic similarity
- **Inner Product**: Requires careful normalization, cosine is more robust
- **BM25**: Keyword-based, not semantic
- **Hybrid Search**: Could be added as future enhancement

## Usage

### Gradio Interface

1. **Enable Reranking**: Check "Use Reranking" checkbox
2. **Set Candidates (m)**: Number of initial candidates (default: 20)
3. **Set Results (n)**: Number of final results (default: 5)
4. **Search**: Enter query and click search

**Example:**
- Query: "How do I solve quadratic equations?"
- m = 20: Retrieve 20 candidates from FAISS
- n = 5: Return top 5 after reranking

### CLI

```bash
# With reranking
python interface/cli.py "quadratic equations" --rerank -m 20 -n 5

# Without reranking (original behavior)
python interface/cli.py "quadratic equations" -k 5
```

### Python API

```python
from src.knowledge_base import KnowledgeBase
from src.embeddings import EmbeddingGenerator
from src.retrieval import RAGRetriever
from src.reranker import BERTReranker

# Initialize
kb = KnowledgeBase('data/sat_knowledge_base.json')
embedder = EmbeddingGenerator()
reranker = BERTReranker()
retriever = RAGRetriever(kb, embedder, reranker=reranker)

# Two-stage retrieval
results = retriever.retrieve_with_reranking(
    query="How do I solve quadratic equations?",
    m=20,  # Retrieve 20 candidates
    n=5    # Return top 5 after reranking
)
```

## Performance Characteristics

### Speed
- **FAISS Stage**: ~10-50ms (depends on knowledge base size)
- **Reranking Stage**: ~50-200ms (depends on m and model)
- **Total**: ~60-250ms per query

### Accuracy
- **Without Reranking**: Good semantic matches, may include some false positives
- **With Reranking**: More accurate relevance scoring, better precision

### Memory
- **Reranker Model**: ~80-100MB (model size)
- **Total System**: ~200-300MB (including embeddings and FAISS index)

## Configuration

### Adjusting m and n

**m (Candidates):**
- **Lower (5-10)**: Faster, but may miss relevant results
- **Higher (30-50)**: Slower, but better recall
- **Recommended**: 20-30 for most use cases

**n (Final Results):**
- **Lower (3-5)**: More focused, faster
- **Higher (10-20)**: More comprehensive, slower
- **Recommended**: 5-10 for most use cases

### Changing Reranker Model

```python
# Use higher quality model (slower)
reranker = BERTReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Use general purpose model
reranker = BERTReranker(model_name="BAAI/bge-reranker-base")
```

## How It Works

### Two-Stage Pipeline

```
Query: "How do I solve quadratic equations?"
    ↓
[Stage 1: FAISS Semantic Search]
    ↓
Retrieve m=20 candidates with similarity scores
    ↓
[Stage 2: BERT Cross-Encoder Reranking]
    ↓
Score each candidate with query-document pair
    ↓
Combine scores (70% rerank + 30% initial)
    ↓
Return top n=5 results
```

### Score Combination

Final score = `0.7 × rerank_score + 0.3 × initial_score`

This gives more weight to the accurate reranker while preserving initial semantic similarity.

## Benefits

1. **Improved Accuracy**: Cross-encoder provides more accurate relevance scoring
2. **Flexible**: Can adjust m and n based on use case
3. **Backward Compatible**: Original `retrieve()` method still works
4. **Configurable**: Easy to enable/disable reranking
5. **Production Ready**: Uses proven models and techniques

## Limitations

1. **Speed**: Reranking adds latency (50-200ms)
2. **Memory**: Additional model requires ~100MB
3. **Model Download**: First run downloads model (~80MB)

## Future Enhancements

Potential improvements:
- [ ] Hybrid search (keyword + semantic)
- [ ] Multiple reranker models (ensemble)
- [ ] Caching rerank results
- [ ] Batch reranking for multiple queries
- [ ] Custom reranker fine-tuning
- [ ] GPU acceleration support

## Troubleshooting

### Reranker Not Loading

If reranker fails to load:
- Check internet connection (model downloads on first use)
- Verify sentence-transformers is installed: `pip install sentence-transformers`
- System falls back to regular retrieval automatically

### Slow Performance

If reranking is too slow:
- Reduce `m` (fewer candidates to rerank)
- Use faster model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Consider disabling reranking for simple queries

### Low Quality Results

If results aren't improving:
- Increase `m` to retrieve more candidates
- Try higher quality model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Check if knowledge base has relevant content

## References

- [Cross-Encoder Models](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
