#!/bin/bash
# run.sh - Entrypoint for the RAG system
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
# Build index if it doesn't exist yet
if [ ! -f "${SCRIPT_DIR}/data/index/faiss.index" ]; then
    python3 "${SCRIPT_DIR}/build.py"
fi
 
python3 "${SCRIPT_DIR}/rag_pipeline.py" "$1" "$2"
