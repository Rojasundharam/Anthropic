import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

def create_embeddings(documents):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(documents)
    return embeddings.toarray()

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_similar(query, index, embeddings, k=5):
    vectorizer = TfidfVectorizer()
    query_embedding = vectorizer.fit_transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_embedding, k)
    return indices[0]