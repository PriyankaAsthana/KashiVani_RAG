def chunk_documents(documents: list[dict], 
                    chunk_size: int = 500, 
                    overlap: int = 50) -> list[dict]:
    
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        filename = doc["filename"]
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "chunk_id": f"{filename}_chunk_{chunk_index}",
                "filename": filename,
                "text": chunk_text,
                "start_char": start,
                "end_char": end
            })
            
            start = end - overlap
            chunk_index += 1
    
    return chunks


if __name__ == "__main__":
    from ingestor import load_pdfs
    
    docs = load_pdfs("data/raw")
    chunks = chunk_documents(docs)
    
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nExample chunk:")
    print(f"ID: {chunks[10]['chunk_id']}")
    print(f"Characters: {chunks[10]['start_char']} to {chunks[10]['end_char']}")
    print(f"Text:\n{chunks[10]['text']}")