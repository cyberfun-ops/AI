import chromadb
from embeddings import create_embedding, RAG_ROOT


# Create a local Chroma database
client = chromadb.PersistentClient(path=str(RAG_ROOT / "chroma_db"))

# Create or load a collection
collection = client.get_or_create_collection(
    name="documents"
)


def search(query, top_k=5, max_distance=0.9):

    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Check the best result
    best_distance = results["distances"][0][0]

    print(f"Best retrieval distance: {best_distance}")

    # If even the best result is too far away,
    # return no results
    if best_distance > max_distance:

        print("No sufficiently relevant information found.")

        return None

    return results




def add_chunk(chunk_id, text, page_number, document_name):

    embedding = create_embedding(text)

    collection.add(
        ids=[chunk_id],
        embeddings=[embedding.tolist()],
        documents=[text],
        metadatas=[{
            "page_number": page_number,
            "document_name": document_name
        }]
    )


if __name__ == "__main__":

    results = search(
        "Why did the company's sales fall?",
        top_k=3
    )

    print("\nSearch results:\n")

    for i, document in enumerate(results["documents"][0]):

        print(f"Result {i + 1}:")
        print(document)

        print(
            "Metadata:",
            results["metadatas"][0][i]
        )

        print()