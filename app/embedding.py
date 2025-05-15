import os
from dotenv import load_dotenv
from mistralai.client import MistralClient

load_dotenv()

class MistralEmbedder:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY not set in environment")
        self.client = MistralClient(api_key=api_key)
        self.model = "mistral-embed" 

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for a list of text strings.

        Args:
            texts (list[str]): A list of text strings.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        response = self.client.embeddings(model=self.model, input=texts)
        return [e.embedding for e in response.data]
