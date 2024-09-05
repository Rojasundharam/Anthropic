import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

class EmbeddingUtil:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def create_embeddings(self, documents):
        embeddings = self.vectorizer.fit_transform(documents)
        return embeddings.toarray()

    def create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def search_similar(self, query, index, embeddings, k=5):
        query_embedding = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = index.search(query_embedding, k)
        return indices[0]

embedding_util = EmbeddingUtil()

def create_embeddings(documents):
    return embedding_util.create_embeddings(documents)

def create_faiss_index(embeddings):
    return embedding_util.create_faiss_index(embeddings)

def search_similar(query, index, embeddings, k=5):
    return embedding_util.search_similar(query, index, embeddings, k)