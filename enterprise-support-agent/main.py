import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Make the Knowledge_RAG modules importable
sys.path.append(str(Path(__file__).resolve().parent / "Knowledge_RAG" / "src"))

from vector_store import search


# LangChain retriever backed by the Knowledge_RAG Chroma store
class MQKnowledgeRetriever(BaseRetriever):
    top_k: int = 5
    max_distance: float = 0.9

    def _get_relevant_documents(self, query, *, run_manager):
        results = search(query, top_k=self.top_k, max_distance=self.max_distance)

        # No sufficiently relevant chunks found
        if results is None:
            return []

        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(
                results["documents"][0],
                results["metadatas"][0]
            )
        ]


def format_docs(docs):
    if not docs:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(
        f"[Source: {doc.metadata['document_name']}, Page {doc.metadata['page_number']}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


retriever = MQKnowledgeRetriever()

# Local Llama model
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=2
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an Enterprise IT Support Assistant.

        Your job is to help engineers troubleshoot
        enterprise technology issues.

        Answer using the context from the IBM MQ
        troubleshooting documentation below. Cite the
        source page for the information you use. If the
        context does not contain the answer, say so
        instead of guessing.

        Give:
        1. A short explanation of the problem
        2. Possible causes
        3. Troubleshooting steps
        4. When to escalate the issue

        Context:
        {context}
        """
    ),
    (
        "human",
        "{question}"
    )
])

# Connect retriever → prompt → LLM
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":

    # Ask a question
    response = chain.invoke(
        "My IBM MQ message is stuck in a queue. What should I check?"
    )

    print(response)
