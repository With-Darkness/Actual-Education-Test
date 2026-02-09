# SAT Knowledge Matching System

A RAG-based (Retrieval-Augmented Generation) system that identifies relevant SAT knowledge points based on student questions or conversations. The system uses semantic search to match student queries with curriculum content across Math, Reading, Writing, and Test-Taking Strategy domains.

## 🎯 Features

- **Semantic Search**: Uses sentence transformers and FAISS for efficient vector similarity search
- **Comprehensive Knowledge Base**: 24+ SAT knowledge points covering major topics
- **Multiple Interfaces**: 
  - Interactive Gradio web interface
  - Command-line interface (CLI)
- **Accurate Retrieval**: Returns top-k most relevant knowledge points with similarity scores
- **Demo Cases**: Includes 5 example queries demonstrating system accuracy

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Demo Cases](#demo-cases)
- [Testing](#testing)
- [Approach & Design Decisions](#approach--design-decisions)

## 🏗️ Architecture

The system follows a modular RAG architecture:

```
Student Query → Embedding Model → Vector Search (FAISS) → Knowledge Base → Ranked Results
```

### Components

1. **Knowledge Base** (`src/knowledge_base.py`)
   - Loads and manages SAT curriculum content from JSON
   - Provides filtering and querying capabilities
   - Converts knowledge points to text for embedding

2. **Embedding Generator** (`src/embeddings.py`)
   - Uses Sentence Transformers (`all-MiniLM-L6-v2`)
   - Generates semantic embeddings for queries and knowledge points
   - Fast and efficient for production use

3. **RAG Retriever** (`src/retrieval.py`)
   - Builds FAISS index for fast similarity search
   - Implements cosine similarity using normalized L2 distance
   - Returns ranked results with similarity scores

4. **Interfaces**
   - **Gradio App**: Web-based UI for interactive queries
   - **CLI**: Command-line tool for programmatic access

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or navigate to project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical operations
- `sentence-transformers` - Embedding generation
- `faiss-cpu` - Vector similarity search
- `gradio` - Web interface
- `python-dotenv` - Environment variable management

3. **Verify knowledge base exists**:
   - Ensure `data/sat_knowledge_base.json` is present
   - The file contains 24 knowledge points across Math, Reading, Writing, and Strategy

## 🚀 Usage

### Option 1: Gradio Web Interface (Recommended)

Launch the interactive web interface:

```bash
python interface/gradio_app.py
```

The interface will start at `http://localhost:7860` (or a different port if 7860 is occupied).

**Features:**
- Enter student questions or topics
- Adjust number of results (1-10)
- View detailed or simplified results
- See system statistics
- Try example queries

### Option 2: Command-Line Interface

#### Basic Search
```bash
python interface/cli.py "How do I solve quadratic equations?"
```

#### Advanced Options
```bash
# Get 10 results in JSON format
python interface/cli.py "subject-verb agreement" -k 10 -f json

# Show knowledge base statistics
python interface/cli.py --stats

# Run demo cases
python interface/cli.py --demo
```

#### CLI Options
- `query`: Student question or topic (positional argument)
- `-k, --top-k`: Number of results (default: 5)
- `-f, --format`: Output format - `markdown`, `text`, or `json` (default: `text`)
- `--kb-path`: Custom path to knowledge base JSON file
- `--demo`: Run demo test cases
- `--stats`: Show knowledge base statistics

### Option 3: Python API

Use the system programmatically:

```python
from src.knowledge_base import KnowledgeBase
from src.embeddings import EmbeddingGenerator
from src.retrieval import RAGRetriever

# Initialize
kb = KnowledgeBase('data/sat_knowledge_base.json')
embedder = EmbeddingGenerator()
retriever = RAGRetriever(kb, embedder)

# Search
results = retriever.retrieve("How do I solve quadratic equations?", top_k=5)

# Process results
for knowledge_point, similarity_score in results:
    print(f"{knowledge_point['topic']}: {similarity_score:.2%}")
```

## 📁 Project Structure

```
sat-knowledge-matching-system/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── data/
│   └── sat_knowledge_base.json    # SAT curriculum knowledge base
│
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py    # Knowledge base loader and manager
│   ├── embeddings.py        # Embedding generation
│   ├── retrieval.py         # RAG retrieval system
│   └── utils.py             # Utility functions
│
├── interface/
│   ├── __init__.py
│   ├── gradio_app.py        # Gradio web interface
│   └── cli.py               # Command-line interface
│
├── examples/
│   └── demo_cases.json      # Demo test cases
│
└── tests/
    ├── __init__.py
    └── test_retrieval.py    # Unit tests
```

## 🎓 Demo Cases

The system includes 5 demo cases demonstrating retrieval accuracy:

1. **"How do I solve quadratic equations?"**
   - Expected: Quadratic Equations, Factoring, Quadratic Formula
   - Category: Math

2. **"What is subject-verb agreement?"**
   - Expected: Subject-Verb Agreement
   - Category: Writing

3. **"I need help understanding the Pythagorean theorem"**
   - Expected: Triangles, Pythagorean Theorem
   - Category: Math

4. **"How can I find the main idea of a reading passage?"**
   - Expected: Main Idea Identification, Reading Comprehension
   - Category: Reading

5. **"What are context clues and how do I use them?"**
   - Expected: Context Clues, Vocabulary
   - Category: Reading

Run demo cases:
```bash
python interface/cli.py --demo
```

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest tests/test_retrieval.py
```

**Test Coverage:**
- Knowledge base loading and filtering
- Embedding generation
- Retrieval accuracy
- Integration tests
- Edge cases (empty queries, thresholds, etc.)

## 🎯 Approach & Design Decisions

### Why RAG?

RAG (Retrieval-Augmented Generation) was chosen because:
- **Accuracy**: Semantic search provides better matching than keyword search
- **Scalability**: Can handle large knowledge bases efficiently
- **Flexibility**: Easy to update knowledge base without retraining
- **Interpretability**: Returns actual knowledge points with similarity scores

### Technical Choices

1. **Embedding Model**: `all-MiniLM-L6-v2`
   - Fast inference (~50ms per query)
   - Good semantic understanding
   - Small model size (~80MB)
   - 384-dimensional embeddings

2. **Vector Database**: FAISS (Facebook AI Similarity Search)
   - Fast similarity search (milliseconds for 24 vectors)
   - Supports cosine similarity via L2 normalization
   - No external dependencies (runs locally)
   - Scales to thousands of knowledge points

3. **Similarity Metric**: Cosine Similarity
   - Better for semantic search than Euclidean distance
   - Normalized scores (0-1 range)
   - Handles varying text lengths well

4. **Knowledge Base Format**: JSON
   - Human-readable and editable
   - Easy to extend with new knowledge points
   - Structured format supports filtering

### Retrieval Process

1. **Indexing Phase** (on first run or when knowledge base changes):
   - Load knowledge points from JSON
   - Generate embeddings for each knowledge point
   - Build FAISS index with normalized vectors
   - **Save index to disk** (`data/faiss_index/index.bin`)
   - **Save metadata** (`data/faiss_index/metadata.json`)

2. **Index Loading** (on subsequent runs):
   - Check if index exists and is up-to-date
   - Validate against knowledge base hash and embedding model
   - Load index from disk if valid (fast startup!)
   - Rebuild if knowledge base changed or index invalid

3. **Query Phase** (per search):
   - Generate embedding for student query
   - Normalize query embedding
   - Search FAISS index for top-k similar vectors
   - Convert distances to similarity scores
   - Return ranked knowledge points

### Knowledge Point Representation

Each knowledge point includes:
- **Topic**: Main concept name
- **Description**: Detailed explanation
- **Key Concepts**: Important points to remember
- **Common Applications**: Where/how it's used
- **Example Problem**: Sample question
- **Example Solution**: Step-by-step solution
- **Difficulty**: Easy/Medium/Hard
- **Related Topics**: Connected concepts

For embedding, we combine: `topic + description + key_concepts + common_applications`

This provides rich semantic information for accurate matching.

## 🔧 Customization

### Adding New Knowledge Points

Edit `data/sat_knowledge_base.json` and add entries to the `knowledge_points` array:

```json
{
  "id": "MATH_013",
  "category": "Math",
  "subcategory": "Algebra",
  "topic": "Polynomials",
  "description": "...",
  "key_concepts": [...],
  "common_applications": [...],
  "example_problem": "...",
  "example_solution": "...",
  "difficulty": "Medium",
  "related_topics": [...]
}
```

The system will automatically detect the change and rebuild the index on next run.
The index is persisted to `data/faiss_index/` for faster subsequent startups.

### Changing Embedding Model

Edit `src/embeddings.py`:

```python
embedder = EmbeddingGenerator(model_name='your-model-name')
```

Popular alternatives:
- `all-mpnet-base-v2` - More accurate, slower
- `paraphrase-MiniLM-L6-v2` - Similar to current, good for paraphrases

### Adjusting Retrieval

Modify `src/retrieval.py`:
- Change similarity threshold in `retrieve_with_threshold()`
- Adjust normalization strategy
- Add re-ranking logic

## 📊 Performance

- **Index Build Time**: ~2-3 seconds (24 knowledge points, first run)
- **Index Load Time**: ~100-200ms (subsequent runs, from disk)
- **Query Time**: ~50-100ms per query
- **Memory Usage**: ~200MB (including model)
- **Index Storage**: ~38 KB on disk (`data/faiss_index/`)
- **Accuracy**: High relevance for semantic queries

### Index Persistence

The FAISS index is automatically saved to disk and loaded on startup:
- **First Run**: Builds index (~2-3 seconds) and saves to `data/faiss_index/`
- **Subsequent Runs**: Loads index from disk (~100-200ms) for fast startup
- **Auto-Update**: Detects knowledge base changes and rebuilds automatically
- **Validation**: Checks hash and model compatibility before loading

See `FAISS_INDEX_PERSISTENCE.md` for detailed documentation.

## 🐛 Troubleshooting

### Model Download Issues
If the embedding model fails to download:
- Check internet connection
- The model downloads automatically on first use (~80MB)
- Models are cached in `~/.cache/torch/sentence_transformers/`

### FAISS Installation Issues
If `faiss-cpu` fails to install:
- Try: `pip install faiss-cpu --no-cache-dir`
- On Apple Silicon: May need `faiss-cpu` or build from source
- On Linux: Usually installs without issues

### Import Errors
If you get import errors:
- Ensure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version: `python --version` (needs 3.8+)

### Index Loading Issues
If the FAISS index fails to load:
- Check that `data/faiss_index/` directory exists and is writable
- Delete `data/faiss_index/` to force a rebuild
- The system will automatically rebuild if the index is invalid or missing
- Check console output for specific error messages

## 📝 License

This project is created for educational and assessment purposes.

## 👤 Author

Created as part of an AI Engineer take-home assignment.

## 🙏 Acknowledgments

- Sentence Transformers library by UKP Lab
- FAISS by Facebook AI Research
- Gradio for the web interface framework
- SAT curriculum content based on public educational resources

---

**Note**: This system is designed as a demonstration of RAG capabilities. For production use, consider:
- Larger, more comprehensive knowledge base
- Fine-tuned embedding models
- Advanced re-ranking techniques
- User feedback integration
- Performance monitoring
