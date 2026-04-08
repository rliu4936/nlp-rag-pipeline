# RAG Question-Answering Pipeline

A retrieval-augmented generation (RAG) system for answering questions about UC Berkeley EECS. Combines hybrid sparse/dense retrieval with LLM-based answer generation to achieve high-accuracy QA over a large unstructured corpus.

## Architecture

```
Question → [Hybrid BM25 + Dense Retriever] → Top-K Passages → [LLM Generator] → Answer
```

The pipeline has three stages:

1. **Retrieval** — A hybrid retriever combines BM25 (lexical) and dense embedding search (semantic) over ~67K text chunks, scoring candidates with a weighted combination to capture both exact keyword matches and meaning-level similarity.

2. **Generation** — Retrieved passages are injected into a structured prompt and sent to an LLM (Qwen 2.5 7B Instruct via OpenRouter) for answer generation.

3. **Evaluation** — Answers are scored against references using Exact Match (EM) and token-level F1.

## Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding model | all-MiniLM-L6-v2 (22M params) | Fast inference, strong retrieval quality |
| Sparse retrieval | BM25 | Reliable lexical matching for entity-heavy queries |
| Dense retrieval | FAISS (L2) | Efficient nearest-neighbor search over embeddings |
| Hybrid weights | BM25 (0.4) + Dense (0.6) | Empirically tuned for best F1 |
| Top-K | 10 passages | Balances context richness vs. prompt length |
| LLM | Qwen 2.5 7B Instruct | Strong instruction-following for extractive QA |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (build index + answer questions + evaluate)
bash run.sh val_QA/val.jsonl data/predictions.txt
```

This will automatically:
1. Build BM25 + FAISS indices if they don't exist
2. Retrieve and generate answers for each question
3. Print EM and F1 scores

### Step-by-Step

```bash
# 1. Build retrieval indices
python3 build.py

# 2. Run RAG pipeline on questions
python3 rag_pipeline.py val_QA/val.jsonl data/predictions.txt

# 3. Evaluate predictions
python3 evaluate.py data/predictions.txt data/predictions_references.json
```

## Project Structure

```
.
├── rag_pipeline.py        # Main RAG pipeline — retrieval + prompt + generation
├── retriever.py           # Hybrid BM25 + dense retriever with FAISS
├── build.py               # Index construction (BM25 + FAISS)
├── llm.py                 # LLM API wrapper (OpenRouter)
├── evaluate.py            # Exact Match + F1 evaluation
├── run.sh                 # One-command entrypoint
├── requirements.txt
├── report.pdf             # Detailed writeup with ablation studies
├── data/
│   ├── eecs_chunks_v3_part_*.jsonl   # Corpus (~67K chunks across 7 files)
│   └── index/                         # Auto-generated retrieval indices
│       ├── bm25.pkl
│       ├── faiss.index
│       └── corpus_cache.pkl
└── val_QA/
    └── val.jsonl          # 102 validation questions with reference answers
```

## Results

Evaluated on 102 validation questions:

| Metric | Score |
|--------|-------|
| Exact Match | See `report.pdf` |
| F1 | See `report.pdf` |

Ablation studies on retrieval strategy (sparse-only, dense-only, hybrid) and top-K values are documented in the report.


