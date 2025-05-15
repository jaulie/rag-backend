import numpy as np
import faiss
import nltk
from nltk.corpus import stopwords
from embedding import MistralEmbedder

class VectorDB:
    def __init__(self, dim: int, embedder: MistralEmbedder):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Flat index using inner product distance
        self.id_to_vector = {} # Map internal IDs to the vector
        self.id_to_text = {} # Map internal IDs to the text itself
        self.current_id = 0  # ID counter
        self.embedder = embedder # We need to re-embed our chunks
        self.stopwords = set(stopwords.words("english")) # Stop words to remove in keyword search

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm
    
    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from the input text.
        """
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)

    def add_vector(self, vector: np.ndarray, text: str):
        """
        Add a vector to the store.

        Args:
            vector (numpy.ndarray): The vector embedding to be stored.
            text (str): The corresponding text.
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
    
    def keyword_search(self, query: str, num_results: int = 5):
        """
        Perform a keyword search on the stored text data.

        Args:
            query (str): The query string containing keywords to search for.
            num_results (int): The number of results to return.

        Returns:
            list: A list of dictionaries with document id, text, and matching score.
        """
        results = []
        query = self._remove_stopwords(query)
        query_words = set(query.lower().split())

        # Search through stored texts
        for idx, text in self.id_to_text.items():
            # Tokenize the stored text and convert to lower case
            text_cleaned = self._remove_stopwords(text)
            document_words = set(text_cleaned.lower().split())
            
            # Find intersection of query words and document words (keywords)
            common_words = query_words.intersection(document_words)
            score = len(common_words)  # Number of common keywords

            if score > 0:
                results.append({
                    "id": idx,
                    "text": text,
                    "score": score
                })
            

        # Sort by score (higher score means more matches)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Return the top 'num_results' results
        return results[:num_results]

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
    
    def hybrid_search(self, query: str, query_vector: np.ndarray, num_results=5, keyword_weight=0.5, semantic_weight=0.5):
        # Get keyword-based results
        keyword_results = self.keyword_search(query, num_results=num_results)
        
        # Get semantic-based results
        semantic_results = self.search(query_vector, num_results=num_results)
        
        # Merge and score the results (this could be improved with more sophisticated ranking strategies)
        merged_results = []
        for idx, keyword_result in enumerate(keyword_results):
            # Merge keyword and semantic scores
            semantic_result = semantic_results[idx] if idx < len(semantic_results) else None
            
            score = (keyword_weight * keyword_result['score']) + (semantic_weight * (semantic_result['score'] if semantic_result else 0))
            
            merged_results.append({
                "id": keyword_result["id"],
                "text": keyword_result["text"],
                "score": score
            })
        
        # Sort by score and return top-k results
        merged_results = sorted(merged_results, key=lambda x: x['score'], reverse=True)
        return merged_results[:num_results]