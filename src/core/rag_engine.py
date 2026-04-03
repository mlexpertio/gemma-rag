from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.vector_store import VectorStoreManager


class RAGEngine:
    def __init__(self, model_name: str, persist_dir: str):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.llm = ChatOllama(model=model_name)
        self.vector_store_manager = VectorStoreManager(
            persist_dir, "qwen3-embedding:0.6b"
        )

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)

    def answer_question(
        self, question: str, md_files: list[str]
    ) -> tuple[str, list[str]]:
        """
        Performs a manual RAG loop: Retrieve -> Augment -> Generate.
        Returns a tuple of (answer, retrieved_chunks).
        """
        self.vector_store_manager.load_or_create(md_files)
        retriever = self.vector_store_manager.get_retriever()

        if not retriever:
            return "No documents loaded to search from.", []

        # 1. Retrieve
        docs = retriever.invoke(question)
        retrieved_chunks = [doc.page_content for doc in docs]

        # 2. Augment
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"Chunk {i + 1}: {chunk}")
        context = "\n\n".join(context_parts)

        if not context:
            return "No relevant context found in the documents.", []

        # 3. Generate
        system_prompt_text = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "IMPORTANT: At the end of your answer, list the numbers of the chunks you used to find the answer (e.g., 'Sources: [1, 3]'). "
            "Note that the chunks are numbered starting from 1 up to the total number of chunks provided in the context.\n\n"
            "CONTEXT:\n"
            "{context}"
        )

        messages = [
            SystemMessage(content=system_prompt_text.format(context=context)),
            HumanMessage(content=question),
        ]

        response = self.llm.invoke(messages)
        return str(response.content), retrieved_chunks
