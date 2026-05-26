# <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=35&duration=3000&pause=1000&color=FF9933&center=true&vCenter=true&width=1000&lines=KashiVani+%F0%9F%95%89%EF%B8%8F;Domain-Specific+RAG+for+Indian+Cultural+Heritage;Semantic+Retrieval+%7C+FAISS+%7C+LLaMA+3.1;Research-Driven+Retrieval-Augmented+Generation" alt="Typing SVG" />

<div align="center">

<img src="https://img.shields.io/badge/Research-NLP-blueviolet?style=for-the-badge"/>
<img src="https://img.shields.io/badge/RAG-Architecture-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LLaMA-3.1_8B-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Sentence_Transformers-Embeddings-success?style=for-the-badge"/>
<img src="https://img.shields.io/github/stars/PriyankaAsthana/KashiVani_RAG?style=for-the-badge"/>
<img src="https://img.shields.io/github/forks/PriyankaAsthana/KashiVani_RAG?style=for-the-badge"/>
<img src="https://img.shields.io/github/license/PriyankaAsthana/KashiVani_RAG?style=for-the-badge"/>

</div>

---

<div align="center">

## ✨ Retrieval-Augmented Generation for Indian Cultural Heritage ✨

**KashiVani** is a modular Retrieval-Augmented Generation (RAG) framework designed for culturally grounded semantic retrieval on Indian heritage corpora.

Built specifically for exploring how different embedding architectures behave under corpus scaling in culturally heterogeneous retrieval environments.

</div>

---

# 🌟 Features

<table>
<tr>
<td width="50%">

### 🔍 Dense Semantic Retrieval
- Sentence Transformer embeddings
- FAISS vector indexing
- Top-k nearest neighbor search
- Exact exhaustive retrieval

### 🧠 Grounded Generation
- LLaMA 3.1 8B Instruct
- Context-constrained answering
- Hallucination reduction
- Deterministic low-temperature generation

</td>
<td width="50%">

### 📚 Cultural Heritage Corpus
- Indian cultural knowledge
- Sanskrit terminology
- Bhojpuri linguistic content
- Temple architecture
- Hindustani classical music
- Religious philosophy

### 📊 Research-Oriented Evaluation
- Source Accuracy metric
- Keyword Accuracy metric
- Cross-scale embedding analysis
- Comparative transformer evaluation

</td>
</tr>
</table>

---

# 🏛️ Architecture

<div align="center">

```mermaid
flowchart TD

    A[📄 Document Ingestion]
    B[✂️ Chunking Strategy]
    C[🧠 Sentence Embeddings]
    D[🗂️ FAISS Vector Index]
    E[🔍 Semantic Retrieval]
    F[🤖 LLaMA 3.1 Generation]
    G[✨ Grounded Answer]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
