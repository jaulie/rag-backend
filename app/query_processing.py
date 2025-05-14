
def is_retrieval_query(query: str, client) -> bool:
    prompt = f"""
    You are an assistant that classifies whether a user query needs document context to be answered.

    Query: "{query}"

    Answer with only "Yes" or "No".
    """
    response = client.chat(prompt=prompt).strip().lower()
    return response.startswith("yes")