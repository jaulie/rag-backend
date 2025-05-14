import numpy as np
import faiss

class VectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Flat index using inner product distance
        self.id_to_vector = {} # Map internal IDs to the vector
        self.id_to_metadata = {}  # Map internal IDs to external IDs or metadata
        self.current_id = 0  # ID counter

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def add_vector(self, external_id: str, vector: np.ndarray):
        """
        Add a vector to the store.

        Args:
            vector_id (str or int): A unique identifier for the vector.
            vector (numpy.ndarray): The vector data to be stored.
        """
        vector = self._normalize(vector).astype('float32')
        self.index.add(np.array([vector]))  # Add to FAISS index
        self.id_to_vector[self.current_id] = vector
        self.id_to_metadata[self.current_id] = external_id
        self.current_id += 1

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store.

        Args:
            vector_id (str or int): The identifier of the vector to retrieve.
        Returns:
            numpy.ndarray: The vector data if found, or None if not found.
        """
        return self.id_to_vector.get(vector_id)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vector_a (np.ndarray): First vector.
            vector_b (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        a = self._normalize(a)
        b = self._normalize(b)
        return float(np.dot(a, b))

    def search(self, query_vector: np.ndarray, num_results: int = 5):
        """
`       Find similar vectors to the query vector using brute-force search.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        query_vector = self._normalize(query_vector).astype('float32')
        D, I = self.index.search(np.array([query_vector]), num_results)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < self.current_id:
                external_id = self.id_to_metadata[idx]
                results.append({"id": external_id, "score": float(score)})
        return results