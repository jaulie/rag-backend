import numpy as np
import faiss
from app.embedding import MistralEmbedder

class VectorDB:
    def __init__(self, dim: int, embedder: MistralEmbedder):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Flat index using inner product distance
        self.id_to_vector = {} # Map internal IDs to the vector
        self.id_to_text = {} # Map internal IDs to the text itself
        self.current_id = 0  # ID counter
        self.embedder = embedder # We need to re-embed our chunks

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def add_vector(self, vector: np.ndarray, text: str):
        """
        Add a vector to the store.

        Args:
            vector (numpy.ndarray): The vector data to be stored.
        """
        vector = self._normalize(vector).astype('float32')
        self.index.add(np.array([vector]))  # Add to FAISS index
        self.id_to_vector[self.current_id] = vector
        self.id_to_text[self.current_id] = text
        self.current_id += 1

    def add_chunks(self, chunks: list[list[str]]):
        """
        Adds multiple chunks to the DB, embedding them first.

        Args:
            chunks (list[list[str]]): The list of chunks to be stored. Each chunk
            is a list of strings.
        """
        # Flatten the chunks
        flattened_chunks = [" ".join(chunk) for chunk in chunks]

        # Embed first
        embeddings = self.embedder.embed_texts(flattened_chunks)

        for vector, text in zip(embeddings, flattened_chunks):
            self.add_vector(vector, text)

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store.

        Args:
            vector_id (str or int): The identifier of the vector to retrieve.
        Returns:
            np.ndarray: The vector data if found, or None if not found.
        """
        return self.id_to_vector.get(vector_id)

    def search(self, query_vector: np.ndarray, num_results: int = 5):
        """
`       Find similar vectors to the query vector.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            List[dict]: Each dict contains the internal ID, the vector, and similarity score.
        """
        query_vector = self._normalize(query_vector).astype('float32')
        D, I = self.index.search(np.array([query_vector]), num_results)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx in self.id_to_vector:
                results.append({
                    "id": idx,
                    "text": self.id_to_text[idx],
                    "score": float(score)
                })

        return results