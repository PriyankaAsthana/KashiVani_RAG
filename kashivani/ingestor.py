import os
from pypdf import PdfReader

def load_pdfs(folder_path: str) -> list[dict]:
    documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            full_text = clean_text(full_text)
            
            documents.append({
                "filename": filename,
                "text": full_text
            })
            
            print(f"Loaded: {filename} — {len(full_text)} characters")
    
    return documents


def clean_text(text: str) -> str:
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    return text


if __name__ == "__main__":
    docs = load_pdfs("data/raw")
    print(f"\nTotal documents loaded: {len(docs)}")
    print(f"\nFirst 300 characters of first document:\n")
    print(docs[0]["text"][:300])