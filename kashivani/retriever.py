import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
INDEX_PATH = "store/index/faiss.index"
METADATA_PATH = "store/index/metadata.json"


def load_index():
    index = faiss.read_index(INDEX_PATH)
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata


def retrieve(query: str, 
             top_k: int = 3) -> list[dict]:
    
    model = SentenceTransformer(MODEL_NAME)
    index, metadata = load_index()
    
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        chunk["distance"] = float(distances[0][i])
        results.append(chunk)
    
    return results


if __name__ == "__main__":
    test_queries = [
        "what is Ganga Aarti?",
        "history of Banarasi silk saree",
        "Buddhist monuments in Sarnath",
        "ghats of Varanasi"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = retrieve(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} — distance: {result['distance']:.4f}")
            print(f"Source: {result['filename']}")
            print(f"Text: {result['text'][:200]}...")