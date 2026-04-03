Build local RAG for arXiv papers. Everything must work entirely locally/private.

The app has 3 main features:

- Upload a PDF and store the contents as a local `md` file
- Chat with a list of selected PDFs (already uploaded)
- Allow the user to select the model to use for the chat

The tech stack:

- LangChain
- Ollama
- Streamlit
- Something that will convert the PDFs to markdown (locally)
- In-memory vector database if needed

Focus is a minimalist, clean, beautiful and easy-to-understand app.