# RAG System for UC Berkeley EECS QA

## CS288 Assignment 3 - Retrieval Augmented Generation

### Architecture

```
Question → [BM25 + Dense Retriever] → Top-K Passages → [LLM Generator] → Answer
```

**Components:**
- **Retriever** (`retriever.py`): Hybrid BM25 + Dense (sentence-transformers/all-MiniLM-L6-v2 + FAISS)
- **Generator** (`rag_pipeline.py`): Prompt construction + LLM call via `llm.py`
- **Evaluation** (`evaluate.py`): Exact Match + F1 metrics
- **Crawler** (`crawler.py`): BFS crawl of eecs.berkeley.edu (not used — pre-built corpus provided)
- **Data Processor** (`data_processor.py`): HTML → clean text chunks (not used — pre-built corpus provided)

### Quick Start

You can run the entire pipeline (build index + answer questions + evaluate) with a single command:

```bash
bash run.sh val_QA/val.jsonl data/predictions.txt
```

This will automatically:
1. Build BM25 + FAISS indices if they don't exist yet
2. Read questions and reference answers from `val.jsonl`
3. Generate predictions via the RAG pipeline
4. Run evaluation (Exact Match + F1)

### Step-by-Step Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Build index (uses pre-built corpus in `data/`):**
```bash
python3 build.py
```

3. **Run on questions:**
```bash
python3 rag_pipeline.py val_QA/val.jsonl data/predictions.txt
```

4. **Evaluate:**
```bash
python3 evaluate.py data/predictions.txt data/predictions_references.json
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Corpus | Pre-built chunked corpus (7 JSONL parts) | Crawling and processing already done |
| Embedding model | all-MiniLM-L6-v2 (22M params) | Fast, effective, well under 400M limit |
| Retrieval | Hybrid BM25 (0.4) + Dense (0.6) | BM25 for exact matches, dense for semantic |
| Top-K | 5 passages | Sufficient context without overwhelming LLM |
| LLM | qwen/qwen-2.5-7b-instruct | Good instruction following for QA |

### File Structure

```
rag_system/
├── run.sh                 # Entrypoint (auto-builds index + runs pipeline + evaluates)
├── build.py               # Build retrieval indices
├── rag_pipeline.py        # Main RAG pipeline
├── retriever.py           # Hybrid BM25 + Dense retriever
├── llm.py                 # LLM API wrapper (DO NOT MODIFY)
├── evaluate.py            # EM + F1 evaluation
├── crawler.py             # Web crawler (not used)
├── data_processor.py      # HTML cleaning and chunking (not used)
├── README.md
├── data/
│   ├── eecs_chunks_v3_part_1.jsonl   # Pre-built corpus (part 1 of 7)
│   ├── eecs_chunks_v3_part_2.jsonl
│   ├── ...
│   ├── eecs_chunks_v3_part_7.jsonl
│   └── index/                         # Auto-generated indices
│       ├── bm25.pkl
│       ├── faiss.index
│       └── corpus_cache.pkl
└── val_QA/
    └── val.jsonl          # Validation questions + answers
```

### Constraints (from assignment)

- Embedding model: ≤ 400MB
- LLM: Must use OpenRouter, allowed models only
- Runtime: ≤ 0.6s per question average
- Environment: Python 3.10.12, 4GB RAM, no GPU
- `llm.py` must not be modified
