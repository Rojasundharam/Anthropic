from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingUtil:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, documents):
        """
        Create embeddings for the input documents.
        :param documents: List of text documents
        :return: A list of embeddings for the input documents
        """
        embeddings = self.model.encode(documents, convert_to_tensor=False)
        return np.array(embeddings)

    def create_faiss_index(self, embeddings):
        """
        Create a FAISS index based on the provided embeddings.
        :param embeddings: List of document embeddings
        :return: FAISS index
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        index.add(embeddings)
        return index

    def search_similar(self, query, index, embeddings, k=5):
        """
        Search for similar embeddings based on the query.
        :param query: Query text
        :param index: FAISS index
        :param embeddings: List of document embeddings
        :param k: Number of top results to return
        :return: List of indices of similar documents
        """
        query_embedding = self.create_embeddings([query])
        _, indices = index.search(query_embedding, k)  # search for k nearest neighbors
        return indices[0]
