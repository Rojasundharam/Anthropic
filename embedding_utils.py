from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingUtil:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change this to another model if needed

    def create_embeddings(self, documents):
        return self.model.encode(documents)

    def create_faiss_index(self, embeddings):
        dim = embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatL2(dim)  # L2 distance
        index.add(embeddings)
        return index

    def search_similar(self, query, index, embeddings, k=3):
        query_embedding = self.model.encode([query])
        distances, indices = index.search(query_embedding, k)  # Find the k most similar documents
        return indices[0]  # Return indices of the top similar documents
