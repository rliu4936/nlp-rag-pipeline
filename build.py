"""
Build script: Build retrieval indices (BM25 + FAISS) from pre-built corpus.

Usage: python3 build.py [--corpus-path PATH]
"""

import os
import argparse
from retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Build the RAG system")
    parser.add_argument("--corpus-path", type=str, default=None,
                        help="Path to corpus file(s), supports glob pattern")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    index_dir = os.path.join(data_dir, "index")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine corpus path
    if args.corpus_path:
        corpus_path = args.corpus_path
    else:
        corpus_path = os.path.join(data_dir, "eecs_chunks_v3_part_*.jsonl")
    
    print(f"Using corpus: {corpus_path}")
    
    # Build indices
    print("=" * 60)
    print("Building retrieval indices")
    print("=" * 60)
    retriever = HybridRetriever(
        corpus_path=corpus_path,
        index_dir=index_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever.build_index()
    
    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"  Corpus: {corpus_path}")
    print(f"  Index:  {index_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
