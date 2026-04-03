import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    def __init__(self, persist_dir: str, embedding_model: str = "nomic-embed-text"):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None

    def load_or_create(self, md_files: list[str]):
        if not md_files:
            return

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

        for file_path in md_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs = text_splitter.create_documents(
                    [text], metadatas=[{"source": os.path.basename(file_path)}]
                )
                documents.extend(docs)

        if documents:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_dir, embedding_function=self.embeddings
            )

    def get_retriever(self, k: int = 4):
        if self.vector_store is None:
            # Try to load from disk if not in memory
            if os.path.exists(self.persist_dir):
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                )
            else:
                return None

        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def clear_store(self):
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
        if os.path.exists(self.persist_dir):
            import shutil

            shutil.rmtree(self.persist_dir)
