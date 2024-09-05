from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingUtil:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, documents):
        """
        Create embeddings for the input documents.
        :param documents: List of text documents
        :return: A list of embeddings for the input documents
        """
        return self.model.encode(documents, convert_to_tensor=False)

def create_faiss_index(embeddings):
    """
    Create a FAISS index based on the provided embeddings.
    :param embeddings: List of document embeddings
    :return: FAISS index
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_similar(query_embedding, index, embeddings):
    """
    Search for similar embeddings based on the query.
    :param query_embedding: Embedding of the query text
    :param index: FAISS index
    :param embeddings: List of document embeddings
    :return: List of indices of similar documents
    """
    D, I = index.search(query_embedding, k=5)
    return I[0]
