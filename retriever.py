"""
Hybrid Retriever: BM25 + Dense retrieval using sentence-transformers and FAISS.
"""

import os
import json
import pickle
import glob
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    """Simple whitespace + lowercased tokenization for BM25."""
    return text.lower().split()


class HybridRetriever:
    def __init__(self, corpus_path="data/corpus.jsonl",
                 index_dir="data/index",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 use_dense=True,
                 use_bm25=True):
        """
        Initialize the hybrid retriever.
        
        Args:
            corpus_path: Path to corpus JSONL file
            index_dir: Directory to save/load index files
            embedding_model_name: Sentence transformer model name (must be <=400M params)
            use_dense: Whether to use dense retrieval
            use_bm25: Whether to use BM25 retrieval
        """
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        self.use_dense = use_dense
        self.use_bm25 = use_bm25
        
        self.corpus = [] # store all the corpus documents
        self.bm25 = None # BM25 index
        self.faiss_index = None #faiss index for dense retrieval
        self.embed_model = None # sentence transformer model
        
        os.makedirs(index_dir, exist_ok=True)
    
    def load_corpus(self):
        """Load the corpus from JSONL file(s). Supports glob patterns like 'data/eecs_chunks_v3_part_*.jsonl'."""
        self.corpus = []
        files = sorted(glob.glob(self.corpus_path))  # support glob pattern
        if not files:
            files = [self.corpus_path]  # fallback: treat as single file
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.corpus.append(json.loads(line))
        print(f"Loaded {len(self.corpus)} chunks from {len(files)} file(s)")
    
    def build_index(self):
        """Build BM25 and FAISS indices."""
        if not self.corpus:
            self.load_corpus()
        
        texts = [doc['text'] for doc in self.corpus] # extract the text field from each document in the corpus to create a list of texts for indexing
        
        # Build BM25 index
        if self.use_bm25:
            print("Building BM25 index...")
            tokenized_corpus = [tokenize(text) for text in texts]
            self.bm25 = BM25Okapi(tokenized_corpus) # create a BM25 index using the tokenized corpus, BM250Okapi is a popular BM25 implementation that takes a list of tokenized documents and builds an index for efficient retrieval
            
            bm25_path = os.path.join(self.index_dir, "bm25.pkl")
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25, f) # save the BM25 index to a file using pickle, which allows us to load it later without rebuilding
            print("BM25 index built and saved.")
        
        # Build FAISS index
        if self.use_dense:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embed_model = SentenceTransformer(self.embedding_model_name)
            
            print("Encoding corpus...")
            embeddings = self.embed_model.encode(
                texts,
                show_progress_bar=True,
                batch_size=128,
                normalize_embeddings=True # normalize the embeddings to unit length, which allows us to use inner product in FAISS as cosine similarity
            )
            embeddings = embeddings.astype(np.float32)
            
            # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
            dim = embeddings.shape[1] # the dimension of the embeddings, which is determined by the sentence transformer model used. For example, all-MiniLM-L6-v2 has 384 dimensions.
            self.faiss_index = faiss.IndexFlatIP(dim) # create a FAISS index for inner prod similarity search
            self.faiss_index.add(embeddings) # add the embeddings to the FAISS index, which allows us to perform efficient similarity search later
            
            faiss_path = os.path.join(self.index_dir, "faiss.index") # save the FAISS index to a file
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"FAISS index built and saved. Dimension: {dim}, Vectors: {embeddings.shape[0]}")
        
        # Save corpus for quick loading
        corpus_cache_path = os.path.join(self.index_dir, "corpus_cache.pkl")
        with open(corpus_cache_path, 'wb') as f:
            pickle.dump(self.corpus, f) # save the corpus to a file, which allows us to load it quickly later without reading the JSONL file again
        
        print("All indices built successfully.")
    
    def load_index(self):
        """Load pre-built indices."""
        # Load corpus cache
        corpus_cache_path = os.path.join(self.index_dir, "corpus_cache.pkl")
        if os.path.exists(corpus_cache_path):
            with open(corpus_cache_path, 'rb') as f:
                self.corpus = pickle.load(f)
        else:
            self.load_corpus()
        
        # Load BM25
        if self.use_bm25:
            bm25_path = os.path.join(self.index_dir, "bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                print("BM25 index loaded.")
            else:
                print("Warning: BM25 index not found, building...")
                texts = [doc['text'] for doc in self.corpus]
                tokenized_corpus = [tokenize(text) for text in texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Load FAISS
        if self.use_dense:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path) # load the FAISS index from the file
                print("FAISS index loaded.")
            
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embed_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded.")
    
    def retrieve(self, query, top_k=5, bm25_weight=0.4, dense_weight=0.6):
        """
        Retrieve top-k documents using hybrid retrieval.
        
        Args:
            query: The question string
            top_k: Number of documents to return
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense retrieval scores
            
        Returns:
            List of (doc, score) tuples
        """
        n = len(self.corpus) # the total number of documents in the corpus
        scores = np.zeros(n) # initialize an array to store the combined scores for each document, which will be updated with BM25 and dense retrieval scores. The final scores will be used to rank the documents and return the top-k results.
        
        # BM25 retrieval
        if self.use_bm25 and self.bm25:
            bm25_scores = self.bm25.get_scores(tokenize(query)) # get the BM25 scores for the query against all documents in the corpus. 
            #The get_scores method takes a tokenized query and returns an array of BM25 scores corresponding to each document in the corpus.
            # Normalize BM25 scores
            max_bm25 = bm25_scores.max()
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25
            scores += bm25_weight * bm25_scores
        
        # Dense retrieval
        if self.use_dense and self.faiss_index and self.embed_model:
            query_embedding = self.embed_model.encode(
                [query],
                normalize_embeddings=True
            ).astype(np.float32)
            
            # Get more candidates for re-ranking
            k_search = min(top_k * 10, n) # search for more candidates to ensure we have enough relevant documents for re-ranking
            similarities, indices = self.faiss_index.search(query_embedding, k_search) 
            # find the top k_search most similar documents to the query embedding in the FAISS index.
            # return the similarity scores and the corresponding indices of the retrieved documents.
            
            # Add dense scores
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0:
                    scores[idx] += dense_weight * max(0, sim)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.corpus[idx], float(scores[idx])))
        
        return results
    
    def retrieve_bm25_only(self, query, top_k=5):
        """Retrieve using BM25 only."""
        return self.retrieve(query, top_k=top_k, bm25_weight=1.0, dense_weight=0.0)
    
    def retrieve_dense_only(self, query, top_k=5):
        """Retrieve using dense retrieval only."""
        return self.retrieve(query, top_k=top_k, bm25_weight=0.0, dense_weight=1.0)


if __name__ == "__main__":
    # Build index
    retriever = HybridRetriever()
    retriever.build_index()
    
    # Test retrieval
    test_queries = [
        "What is the office number of Dan Klein?",
        "Which year did Dan Klein receive the Grace Murray Hopper Award?",
    ]
    
    retriever.load_index()
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        for doc, score in results:
            print(f"  Score: {score:.4f} | URL: {doc['url']}")
            print(f"  Text: {doc['text'][:200]}...")
