
import sys

from ollama import embeddings
sys.path.append("src")

from document_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embedding

import chromadb


# Create/load our local Chroma database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def ingest_document(file_path, document_name):

    # 1. Load PDF
    pages = load_pdf(file_path)

    print(f"Loaded {len(pages)} pages.")

    chunk_count = 0

    # 2. Process every page
    for page in pages:

        # 3. Split page into chunks
        chunks = chunk_text(page["text"])

        for chunk in chunks:

            # Ignore empty chunks
            if not chunk.strip():
                continue

            # 4. Create embedding
            embedding = create_embedding(chunk)

            # 5. Create unique ID
            chunk_id = f"{document_name}-page-{page['page_number']}-chunk-{chunk_count}"

            # 6. Store everything in Chroma
            print(embedding[1:3])
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                
                documents=[chunk],
                metadatas=[{
                    "page_number": page["page_number"],
                    "document_name": document_name
                }]
            )

            chunk_count += 1

            print(f"Processed chunk {chunk_count}")

    print("\nIngestion complete!")
    print(f"Total chunks stored: {chunk_count}")


if __name__ == "__main__":

    ingest_document(
        "data/documents/mq94.troubleshoot.pdf",
        "mq94.troubleshoot.pdf"
    )

