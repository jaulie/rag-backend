import re
import numpy as np
from embedding import MistralEmbedder

def chunk_sentences(text: str) -> list[str]:
    """
    Chunk text into sentences.
    """
    sentences = re.split(r'(?<=\.)\s+', text)
    return sentences

class SemanticChunker:
    def __init__(self, embedder: MistralEmbedder):
        self.embedder = embedder

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Returns 1 - cosine similarity."""
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - similarity

    def chunk(self, paragraphs: list[str]) -> list[list[str]]:
        """
        Call semantic chunking for each paragraph.

        Args:
            paragraphs: List of paragraphs.

        Returns:
            A list of chunks (each a list of strings).
        """
        chunks = []
        for paragraph in paragraphs:
            sentences = chunk_sentences(paragraph)
            embeddings = self.embedder.embed_texts(sentences)
            semantic_chunks = self.semantic_chunk(sentences, embeddings)
            chunks.extend(semantic_chunks)
        
        return chunks

    def semantic_chunk(self, sentences: list[str], embeddings: list[list[float]], threshold_std: float = 1.0) -> list[list[str]]:
        """
        Perform semantic chunking based on differences in embeddings.

        Args:
            sentences: list of sentences.
            embeddings: list of embeddings
            threshold_std: The number of standard deviations above the mean to consider a semantic shift.

        Returns:
            A list of chunks (each a list of strings).
        """
        if not sentences:
            return []

        # Nothing to chunk if there is only one sentence
        if len(embeddings) <= 1:
            return [sentences]
        
        # Compute semantic distance between adjacent segments
        diffs = [self._cosine_distance(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # Determine dynamic threshold using standard deviation
        threshold = mean_diff + threshold_std * std_diff

        # Split into chunks
        chunks = []
        current_chunk = [sentences[0]]
        for i in range(1, len(sentences)):
            if diffs[i - 1] > threshold:
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(sentences[i])
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
