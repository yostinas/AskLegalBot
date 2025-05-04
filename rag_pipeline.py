from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and prepare data
def load_dataset(path="legal_dataset.txt"):
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs

documents = load_dataset()
doc_embeddings = model.encode(documents)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def answer_query(question):
    question_embedding = model.encode([question])
    _, indices = index.search(np.array(question_embedding), k=1)
    best_match = documents[indices[0][0]]
    return best_match