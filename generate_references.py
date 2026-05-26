import json
from kashivani.ingestor import load_pdfs
from kashivani.chunker import chunk_documents
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def find_relevant_text(question: str, 
                        expected_source: str, 
                        chunks: list[dict]) -> str:
    source_chunks = [
        c for c in chunks 
        if c["filename"] == expected_source
    ]
    
    question_words = set(question.lower().split())
    
    best_chunk = ""
    best_score = 0
    
    for chunk in source_chunks:
        chunk_words = set(chunk["text"].lower().split())
        score = len(question_words & chunk_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk["text"]
    
    return best_chunk


def generate_reference(question: str, context: str) -> str:
    prompt = f"""Based ONLY on the following context, write a concise 
factual answer to the question in 1-3 sentences.
Do not use any outside knowledge.

Context: {context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Loading documents and chunks...")
    docs = load_pdfs("data/raw")
    chunks = chunk_documents(docs)
    
    with open("data/eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    print(f"Generating reference answers for {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        context = find_relevant_text(
            q["question"], 
            q["expected_source"], 
            chunks
        )
        
        reference = generate_reference(q["question"], context)
        q["reference"] = reference
        
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")
        print(f"  → {reference[:80]}...")
    
    with open("data/eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print("\nDone. Reference answers added to eval_questions.json")