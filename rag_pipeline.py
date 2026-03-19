"""
Main RAG pipeline for UC Berkeley EECS QA.
Takes questions as input, retrieves relevant passages, and generates answers.
"""

import sys
import os
import json
import time
from retriever import HybridRetriever
from llm import call_llm, DEFAULT_MODEL


# Configuration
TOP_K = 7  # Number of passages to retrieve
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
MAX_TOKENS = 50 # Max tokens for the answer (short and concise)
TEMPERATURE = 0.0 # Deterministic answers


def build_prompt(question, passages):
    """
    Build the prompt for the LLM given a question and retrieved passages.
    Returns (system_prompt, query) tuple for the new llm.py interface.
    """
    # passages is a list of tuples: (doc, score)
    # Concatenate the documents with the question
    context_parts = []
    for i, (doc, score) in enumerate(passages, 1):
        title = doc.get('title', 'Unknown') # get the title if available, else 'Unknown'
        text = doc['text'] # get the passage text
        # Truncate long passages
        if len(text) > 800:
            text = text[:800] + "..."
        context_parts.append(f"[Passage {i}] (Source: {title})\n{text}")
        # [Passage 1] (Source: Dan Klein Faculty Page) Dan Klein is a professor in the CS department...
    
    context = "\n\n".join(context_parts)
    
    system_prompt = "You are a helpful assistant that answers questions about UC Berkeley EECS based on the provided passages. Answer the question concisely with just the answer itself — no explanation needed. If the answer is a name, number, date, or short phrase, just give that. If the question is a Yes/No question, answer with just \"Yes\" or \"No\"."
    
    query = f"""Context passages:
{context}

Question: {question}

Answer (short, concise, no explanation):"""
    
    return system_prompt, query


def answer_question(question, retriever):
    """
    Answer a single question using the RAG pipeline.
    
    Args:
        question: The question string
        retriever: HybridRetriever instance
    
    Returns:
        Answer string
    """
    # Retrieve relevant passages
    passages = retriever.retrieve(question, top_k=TOP_K)
    
    if not passages:
        # Fallback: try to answer without context
        system_prompt = ""
        query = f"Answer this question about UC Berkeley EECS concisely: {question}\nAnswer:"
    else:
        system_prompt, query = build_prompt(question, passages)
    
    # Call LLM
    try:
        answer = call_llm(
            query=query,
            system_prompt=system_prompt,
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
    except Exception as e:
        print(f"LLM error for question '{question[:50]}...': {e}")
        answer = ""
    
    # Clean answer
    answer = answer.strip() # Remove leading/trailing whitespace
    # Remove any newlines within the answer
    answer = answer.replace('\n', ' ').strip() # Replace newlines with space and trim again
    # Remove common prefixes the LLM might add
    for prefix in ["Answer:", "The answer is", "Based on the passages,"]: # Common prefixes to remove
        if answer.lower().startswith(prefix.lower()): # Remove the prefix if it exists
            answer = answer[len(prefix):].strip() # Trim again after removing prefix
    # Remove trailing period if it's a short answer
    if answer.endswith('.') and len(answer.split()) <= 10:
        answer = answer[:-1].strip()
    
    return answer if answer else "unknown"


def main():
    """
    Main entry point. 
    Usage: python3 rag_pipeline.py <questions_path> <output_path>
    """
    if len(sys.argv) != 3: # Check for correct number of arguments: sys.argv = ['rag_pipeline.py', 'questions.txt', 'predictions.txt']
        print("Usage: python3 rag_pipeline.py <questions_txt_path> <predictions_out_path>")
        sys.exit(1)
    
    questions_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load questions (supports .jsonl with "question" field, or plain text one per line)
    references = []  # store ground truth answers if available
    questions = []
    if questions_path.endswith('.jsonl'):
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    questions.append(item['question'])
                    if 'answer' in item:
                        references.append(item['answer'])
    else:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]
            # Remove trailing empty line if file ends with newline
            while questions and questions[-1] == '':
                questions.pop()
    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize retriever
    print("Loading retriever...")
    # Determine the script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
    corpus_path = os.path.join(script_dir, "data", "eecs_chunks_v3_part_*.jsonl")
    index_dir = os.path.join(script_dir, "data", "index")
    
    retriever = HybridRetriever(
        corpus_path=corpus_path,
        index_dir=index_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_dense=True,
        use_bm25=True
    )
    retriever.load_index()
    print("Retriever loaded.")
    
    # Answer questions
    answers = []
    start_time = time.time() # Start the timer for the entire question-answering process
    
    for i, question in enumerate(questions):
        q_start = time.time() # Start timer for this question
        
        try:
            answer = answer_question(question, retriever)
        except Exception as e:
            print(f"Error on question {i+1}: {e}")
            answer = "unknown"
        
        answers.append(answer)
        
        elapsed = time.time() - q_start # Time taken for this question
        if (i + 1) % 10 == 0: # Print progress every 10 questions
            total_elapsed = time.time() - start_time # Total time elapsed so far
            avg_time = total_elapsed / (i + 1) # Average time per question so far
            print(f"  [{i+1}/{len(questions)}] avg: {avg_time:.2f}s/q, "
                  f"last: {elapsed:.2f}s")
    
    total_time = time.time() - start_time # Total time taken for all questions
    print(f"\nTotal time: {total_time:.1f}s, Avg: {total_time/len(questions):.2f}s/q")
    
    # Write predictions
    with open(output_path, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(answer + '\n')
    
    print(f"Predictions written to {output_path}")
    
    # If reference answers were loaded from jsonl, save them for evaluation
    if references:
        ref_path = output_path.replace('.txt', '') + '_references.json'
        with open(ref_path, 'w', encoding='utf-8') as f:
            json.dump(references, f, ensure_ascii=False, indent=2)
        print(f"Reference answers written to {ref_path}")


if __name__ == "__main__":
    main()
