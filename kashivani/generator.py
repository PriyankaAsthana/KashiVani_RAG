import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['filename']} ---\n"
        context += chunk["text"]
        context += "\n"
    
    prompt = f"""You are Kashivani, a knowledgeable guide to the ancient city of Varanasi.
Answer the user's question using ONLY the information provided in the sources below.
If the answer is not in the sources, say "I don't have enough information about that in my knowledge base."
Do not use any outside knowledge. Be precise and cite which source you used.

SOURCES:
{context}

USER QUESTION: {query}

ANSWER:"""
    
    return prompt


def generate_answer(query: str, chunks: list[dict]) -> str:
    prompt = build_prompt(query, chunks)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    from retriever import retrieve
    
    test_queries = [
        "what is Ganga Aarti?",
        "what are the main ghats of Varanasi?",
        "what is special about Banarasi silk?"
    ]
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("=" * 60)
        
        chunks = retrieve(query, top_k=3)
        answer = generate_answer(query, chunks)
        
        print(answer)
        print()