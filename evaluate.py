import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from kashivani.ingestor import load_pdfs
from kashivani.chunker import chunk_documents
from kashivani.embedder import build_index
import os
import shutil

MODELS_TO_TEST = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2"
]

EVAL_PATH = "data/eval_questions.json"
TOP_K = 3


def build_index_for_model(model_name: str, chunks: list[dict]) -> tuple:
    print(f"\nBuilding index for: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return model, index


def evaluate_model(model_name: str,
                   model,
                   index,
                   chunks: list[dict],
                   questions: list[dict]) -> dict:

    correct_source = 0
    correct_keywords = 0
    total = len(questions)
    results = []

    for q in questions:
        query = q["question"]
        expected_source = q["expected_source"]
        expected_keywords = q["expected_keywords"]

        query_vector = model.encode([query])
        query_vector = np.array(query_vector).astype("float32")

        distances, indices = index.search(query_vector, TOP_K)

        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        retrieved_texts = " ".join([c["text"] for c in retrieved_chunks])
        retrieved_sources = [c["filename"] for c in retrieved_chunks]

        source_hit = expected_source in retrieved_sources

        keyword_hits = sum(
            1 for kw in expected_keywords
            if kw.lower() in retrieved_texts.lower()
        )
        keyword_score = keyword_hits / len(expected_keywords)

        if source_hit:
            correct_source += 1
        if keyword_score >= 0.5:
            correct_keywords += 1

        results.append({
            "question": query,
            "source_hit": source_hit,
            "keyword_score": round(keyword_score, 2),
            "retrieved_sources": retrieved_sources
        })

    return {
        "model": model_name,
        "source_accuracy": round(correct_source / total * 100, 1),
        "keyword_accuracy": round(correct_keywords / total * 100, 1),
        "detailed_results": results
    }


if __name__ == "__main__":
    print("Loading documents and chunks...")
    docs = load_pdfs("data/raw")
    chunks = chunk_documents(docs)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Evaluating {len(questions)} questions across {len(MODELS_TO_TEST)} models...")

    all_results = []

    for model_name in MODELS_TO_TEST:
        model, index = build_index_for_model(model_name, chunks)
        result = evaluate_model(model_name, model, index, chunks, questions)
        all_results.append(result)

        print(f"\nModel: {model_name}")
        print(f"  Source accuracy:  {result['source_accuracy']}%")
        print(f"  Keyword accuracy: {result['keyword_accuracy']}%")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<45} {'Source Acc':>10} {'Keyword Acc':>12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model']:<45} {r['source_accuracy']:>9}% {r['keyword_accuracy']:>11}%")

    with open("data/eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDetailed results saved to data/eval_results.json")