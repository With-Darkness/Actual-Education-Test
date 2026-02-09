# FAISS Index Persistence Implementation

## Overview

The FAISS index is now persisted to disk, allowing for faster startup times and automatic updates when the knowledge base changes.

## How It Works

### Index Storage

The FAISS index is stored in:
```
data/faiss_index/
├── index.bin          # FAISS index binary file
└── metadata.json      # Index metadata (hash, model info, etc.)
```

### Startup Flow

1. **Check for Existing Index**
   - Looks for `data/faiss_index/index.bin` and `metadata.json`
   - If not found → Build new index

2. **Validate Index**
   - Checks knowledge base hash (detects content changes)
   - Checks embedding model (detects model changes)
   - Checks index size matches knowledge base
   - If invalid → Rebuild index

3. **Load or Build**
   - If valid → Load index from disk (fast!)
   - If invalid/missing → Build new index and save

### Change Detection

The system uses a hash-based approach to detect changes:

- **Knowledge Base Hash**: SHA256 hash of:
  - Number of knowledge points
  - Sorted list of knowledge point IDs
  - Knowledge base file path
  - File modification time

- **Model Check**: Compares current embedding model with stored model

If either changes, the index is automatically rebuilt.

## Benefits

1. **Faster Startup**: Loading index (~100ms) vs building (~2-3 seconds)
2. **Automatic Updates**: Detects knowledge base changes and rebuilds
3. **Model Safety**: Rebuilds if embedding model changes
4. **Persistent Storage**: Index survives restarts

## Usage

### Automatic (Default Behavior)

The system automatically handles index persistence:

```python
from src.retrieval import RAGRetriever
from src.knowledge_base import KnowledgeBase
from src.embeddings import EmbeddingGenerator

# First run: Builds and saves index
retriever = RAGRetriever(kb, embedder)

# Subsequent runs: Loads index from disk
retriever = RAGRetriever(kb, embedder)  # Fast startup!
```

### Manual Index Update

If you modify the knowledge base programmatically:

```python
# After modifying knowledge base
retriever.update_index()  # Force rebuild and save
```

### Custom Index Directory

```python
retriever = RAGRetriever(
    kb, 
    embedder, 
    index_dir="custom/path/to/index"
)
```

## File Structure

### `index.bin`
- FAISS binary index file
- Contains normalized embeddings
- Size: ~(num_vectors × dimension × 4 bytes)

### `metadata.json`
```json
{
  "knowledge_base_hash": "abc123...",
  "embedding_model": "all-MiniLM-L6-v2",
  "num_vectors": 24,
  "dimension": 384,
  "index_type": "IndexFlatL2",
  "knowledge_base_path": "data/sat_knowledge_base.json"
}
```

## When Index is Rebuilt

The index is automatically rebuilt when:

1. ✅ Index file doesn't exist (first run)
2. ✅ Knowledge base content changes (hash mismatch)
3. ✅ Embedding model changes
4. ✅ Index size doesn't match knowledge base
5. ✅ Index file is corrupted

## Performance

### First Run (Build)
- Time: ~2-3 seconds
- Actions: Generate embeddings, build index, save to disk

### Subsequent Runs (Load)
- Time: ~100-200ms
- Actions: Load index from disk, validate

### After Knowledge Base Update
- Time: ~2-3 seconds (rebuild)
- Actions: Detect change, rebuild, save

## Troubleshooting

### Index Not Loading

If index fails to load:
1. Check `data/faiss_index/` directory exists
2. Verify file permissions
3. Check error messages in console
4. System will automatically rebuild if needed

### Force Rebuild

To force a rebuild, delete the index directory:
```bash
rm -rf data/faiss_index/
```

Or delete just the files:
```bash
rm data/faiss_index/index.bin
rm data/faiss_index/metadata.json
```

### Index Size

For 24 knowledge points with 384-dimensional embeddings:
- Index size: ~37 KB
- Metadata size: ~500 bytes
- Total: ~38 KB

## Git Integration

The `data/faiss_index/` directory is already in `.gitignore`, so index files won't be committed to version control. This is correct because:
- Index files are generated from knowledge base
- Index files are machine-specific
- Index files can be large for big knowledge bases

## Example Workflow

```python
# Day 1: First run
retriever = RAGRetriever(kb, embedder)
# Output: "Building FAISS index..."
# Output: "Index built and saved successfully!"

# Day 2: Second run (knowledge base unchanged)
retriever = RAGRetriever(kb, embedder)
# Output: "Loading FAISS index from data/faiss_index/index.bin..."
# Output: "Index loaded successfully! (24 vectors)"

# Day 3: Knowledge base updated
# (You add new knowledge points to JSON)
retriever = RAGRetriever(kb, embedder)
# Output: "Knowledge base has changed (hash mismatch), rebuilding index..."
# Output: "Index built and saved successfully!"
```

## Technical Details

### Hash Generation

The hash includes:
- Knowledge point count
- Sorted IDs (for stable ordering)
- File path (for multi-file scenarios)
- Modification time (for file change detection)

This ensures the index is rebuilt when:
- Points are added/removed
- Points are modified
- File is replaced

### Index Validation

Before loading, the system validates:
1. Index file exists and is readable
2. Metadata file exists and is valid JSON
3. Hash matches current knowledge base
4. Model matches current embedding model
5. Index size matches knowledge base size

If any check fails, the index is rebuilt.

## Future Enhancements

Potential improvements:
- [ ] Incremental updates (add/remove single vectors)
- [ ] Index versioning
- [ ] Compression for large indices
- [ ] Distributed index storage
- [ ] Index backup/restore
