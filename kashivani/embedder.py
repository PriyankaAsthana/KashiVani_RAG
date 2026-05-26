import os
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-mpnet-base-v2"
INDEX_PATH = "store/index/faiss.index"
METADATA_PATH = "store/index/metadata.json"


def build_index(chunks: list[dict]) -> None:
    os.makedirs("store/index", exist_ok=True)
    
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, 
                              show_progress_bar=True,
                              batch_size=32)
    
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")
    
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "index": i,
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"]
        })
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to {METADATA_PATH}")
    print(f"\nIndex built successfully — {index.ntotal} vectors stored")


if __name__ == "__main__":
    from ingestor import load_pdfs
    from chunker import chunk_documents
    
    docs = load_pdfs("data/raw")
    chunks = chunk_documents(docs)
    build_index(chunks)
