from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np


# Resolve paths from this file so imports work from any working directory
RAG_ROOT = Path(__file__).resolve().parent.parent

# Load the embedding model
model = SentenceTransformer(
    str(RAG_ROOT / "models" / "all-MiniLM-L6-v2"),
    local_files_only=True
)


def create_embedding(text):
    embedding = model.encode(text)

    return embedding


def cosine_similarity(vector1, vector2):

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


if __name__ == "__main__":

    text1 = "Revenue declined because of weak demand."
    text2 = "Sales fell because customers were buying fewer products."
    text3 = "The company opened a new office in Mumbai."

    embedding1 = create_embedding(text1)
    embedding2 = create_embedding(text2)
    embedding3 = create_embedding(text3)

    print("Embedding dimensions:", len(embedding1))

    print("\nText 1 vs Text 2:")
    print(cosine_similarity(embedding1, embedding2))

    print("\nText 1 vs Text 3:")
    print(cosine_similarity(embedding1, embedding3))