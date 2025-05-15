import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = MistralClient(api_key=api_key)

def is_retrieval_query(query: str) -> bool:
    """
    Invokes the model to check if retrieval is necessary for the query
    """

    prompt = f"""
    You are an assistant that classifies whether a user query needs document context to be answered.

    Query: "{query}"

    Answer with only "Yes" or "No".
    """
    chat_response = client.chat(
        model=model,
        messages=[
            ChatMessage(role="user", content=prompt)
        ]
    )
    response = chat_response.choices[0].message.content.strip().lower()
    return response.startswith("yes")

def run(query: str):
    """
    Invoke Mistral model with just a query as part of the user message
    """
    chat_response = client.chat(
        model=model,
        messages=[
            ChatMessage(role="user", content=query)
        ]
    )
    return (chat_response.choices[0].message.content)

def run_with_context(query: str, context: str):
    """
    Invoke Mistral model with both context and query as part of the user message
    """
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """

    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    chat_response = client.chat(
        model=model,
        messages=[
            ChatMessage(role="user", content=prompt)
        ]
    )
    return (chat_response.choices[0].message.content)