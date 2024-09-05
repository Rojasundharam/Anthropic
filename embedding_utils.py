import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingUtil:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Use any relevant model

    def create_embeddings(self, documents):
        embeddings = self.model.encode(documents)
        return embeddings

    def create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]  # Embedding dimension
        index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
        index.add(np.array(embeddings))  # Add document embeddings to the index
        return index

    def search_similar(self, query, index, embeddings):
        query_embedding = self.model.encode([query])  # Create embedding for the query
        _, indices = index.search(query_embedding, k=5)  # Find top 5 similar documents
        return indices[0]
