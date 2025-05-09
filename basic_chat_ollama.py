import requests
import json

# Ollama server URL (default)
OLLAMA_URL = "http://localhost:11434/api/chat"  # Or your custom URL

def chat_with_ollama(model="granite3.3:8b", messages=None, temperature=0.7, max_tokens=150):
    """
    Sends a chat request to the Ollama server.

    Args:
        model (str): The name of the Ollama model to use.
        messages (list): A list of message dictionaries (see Ollama API docs).
        temperature (float): Controls randomness (0 to 1).
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The model's response.
    """

    if messages is None:
        messages = [{"role": "user", "content": "Hello, Ollama."}]  # Default message

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # Set to True for streaming responses
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about its history."}
    ]

    response = chat_with_ollama(model="granite3.3:8b", messages=conversation)

    if response:
        print(response)

    response = chat_with_ollama(model="granite3.3:8b", messages=[{"role": "user", "content": "Write a short Python function to calculate the area of a rectangle."}])
    if response:
        print(response)