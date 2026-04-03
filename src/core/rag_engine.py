import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.vector_store import VectorStoreManager


class RAGEngine:
    def __init__(self, model_name: str, persist_dir: str):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.llm = ChatOllama(model=model_name)
        self.vector_store_manager = VectorStoreManager(persist_dir, model_name)

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)

    def answer_question(self, question: str, md_files: list[str]) -> str:
        """
        Performs a manual RAG loop: Retrieve -> Augment -> Generate.
        """
        self.vector_store_manager.load_or_create(md_files)
        retriever = self.vector_store_manager.get_retriever()

        if not retriever:
            return "No documents loaded to search from."

        # 1. Retrieve
        docs = retriever.invoke(question)

        # 2. Augment
        context = "\n\n".join([doc.page_content for doc in docs])

        if not context:
            return "No relevant context found in the documents."

        # 3. Generate
        system_prompt_text = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "CONTEXT:\n"
            "{context}"
        )

        messages = [
            SystemMessage(content=system_prompt_text.format(context=context)),
            HumanMessage(content=question),
        ]

        response = self.llm.invoke(messages)
        return str(response.content)
