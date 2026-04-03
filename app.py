import os

import ollama
import streamlit as st

from src.core.rag_engine import RAGEngine
from src.utils.pdf_processor import convert_pdf_to_md

# Configuration
PAPERS_DIR = "papers"
MD_DIR = "papers_md"
VECTOR_STORE_DIR = "vector_store"

st.set_page_config(page_title="Local arXiv RAG", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "llama3"  # Default fallback

# Ensure directories exist
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)


def get_ollama_models():
    try:
        models_info = ollama.list()
        return [m["model"] for m in models_info.get("models", [])]
    except Exception:
        return ["llama3", "mistral", "phi3"]


def get_available_papers():
    papers = []
    if os.path.exists(PAPERS_DIR):
        for f in os.listdir(PAPERS_DIR):
            if f.endswith(".pdf"):
                papers.append(f)
    return papers


def get_available_md_files():
    md_files = []
    if os.path.exists(MD_DIR):
        for f in os.listdir(MD_DIR):
            if f.endswith(".md"):
                md_files.append(f)
    return md_files


# --- Sidebar ---
st.sidebar.title("⚙️ Settings")

# Model Selection
available_models = get_ollama_models()
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=available_models,
    index=available_models.index(st.session_state.current_model)
    if st.session_state.current_model in available_models
    else 0,
)
st.session_state.current_model = selected_model

st.sidebar.divider()

# PDF Upload
st.sidebar.title("📁 Paper Management")
uploaded_file = st.sidebar.file_uploader("Upload arXiv PDF", type="pdf")

if uploaded_file:
    pdf_path = os.path.join(PAPERS_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar.status("Processing PDF...", expanded=False) as status:
        st.write("Converting to Markdown...")
        md_path = convert_pdf_to_md(pdf_path, MD_DIR)
        st.write(f"Done: {os.path.basename(md_path)}")
        status.update(label="Processing Complete!", state="complete", expanded=False)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

st.sidebar.divider()

# Paper Selection for Chat
st.sidebar.title("📚 Selected Papers")
all_md_files = get_available_md_files()
selected_papers = []

if not all_md_files:
    st.sidebar.info("No papers uploaded yet.")
else:
    for md_file in all_md_files:
        # Display original PDF name in checkbox if possible
        display_name = md_file.replace(".md", "")
        if st.sidebar.checkbox(display_name, key=md_file):
            selected_papers.append(os.path.join(MD_DIR, md_file))

# --- Main Chat Area ---
st.title("🧠 arXiv Local RAG")
st.caption(f"Using Model: **{st.session_state.current_model}**")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask something about your selected papers..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare RAG
    if not selected_papers:
        response_text = "⚠️ Please select at least one paper from the sidebar to chat."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    engine = RAGEngine(
                        model_name=st.session_state.current_model,
                        persist_dir=VECTOR_STORE_DIR,
                    )
                    response_text = engine.answer_question(prompt, selected_papers)
                except Exception as e:
                    response_text = f"❌ Error: {str(e)}"
            st.markdown(response_text)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
