import os
from sentence_transformers import SentenceTransformer
from kashivani.ingestor import load_pdfs
from kashivani.chunker import chunk_documents
from kashivani.embedder import build_index, INDEX_PATH, METADATA_PATH
from kashivani.retriever import load_index, retrieve
from kashivani.generator import generate_answer
import numpy as np
import faiss
import json

MODEL_NAME = "all-MiniLM-L6-v2"


class KashivaniPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        
        print("Loading FAISS index...")
        self.index, self.metadata = load_index()
        
        print("Kashivani ready.\n")
    
    def ask(self, query: str, top_k: int = 3) -> dict:
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        
        distances, indices = self.index.search(query_vector, top_k)
        
        chunks = []
        for i, idx in enumerate(indices[0]):
            chunk = self.metadata[idx].copy()
            chunk["distance"] = float(distances[0][i])
            chunks.append(chunk)
        
        answer = generate_answer(query, chunks)
        
        return {
            "query": query,
            "answer": answer,
            "sources": chunks
        }


if __name__ == "__main__":
    pipeline = KashivaniPipeline()
    
    while True:
        query = input("Ask Kashivani: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        result = pipeline.ask(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  - {s['filename']} (distance: {s['distance']:.4f})")
        print()