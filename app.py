import streamlit as st
from pipeline import KashivaniPipeline

st.set_page_config(
    page_title="Kashivani",
    page_icon="🪔",
    layout="centered"
)

st.title("🪔 Kashivani")
st.caption("The Voice of Kashi — A RAG-powered guide to Varanasi")

@st.cache_resource
def load_pipeline():
    return KashivaniPipeline()

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about Varanasi..."):
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = pipeline.ask(query)
        
        st.markdown(result["answer"])
        
        with st.expander("Sources retrieved"):
            for i, source in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1}** — `{source['filename']}` (distance: `{source['distance']:.4f}`)")
                st.markdown(f"> {source['text'][:300]}...")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })