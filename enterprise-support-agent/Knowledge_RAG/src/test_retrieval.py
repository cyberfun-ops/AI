import sys
sys.path.append("src")

from vector_store import search

if __name__ == "__main__":
    question = "What was the revenue from operations in FY2026?"

    results = search(question, top_k=5)

    print("\n===== RETRIEVED CHUNKS =====\n")

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"--- Result {i + 1} ---")

        print("Page:", metadata["page_number"])
        print("Document:", metadata["document_name"])
        print("Distance:", distance)

        print("\nText:")
        print(document)

        print("\n")

