import sys
sys.path.append("src")

from vector_store import search
from generator import generate_answer


def ask_question(question):
    results = search(question, top_k=5)

    # No relevant information found
    if results is None:
        return "I don't have enough information in the document to answer this."

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i in range(len(documents)):
        page = metadatas[i]["page_number"]
        document = metadatas[i]["document_name"]
        text = documents[i]

        context_parts.append(
            f"[Source: {document}, Page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    answer = generate_answer(question, context)

    return answer


if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = ask_question(question)

    print("\nAnswer:")
    print(answer)