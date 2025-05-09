from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Make sure this matches your Ollama model name
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.7,
    max_tokens=800,  # Equivalent to max_new_tokens in the original code
)

def generate_with_length_control(prompt, length="short"):
    """
    Generate a response with length control using Granite model via Ollama.
    
    Args:
        prompt: The prompt/question to send to the model
        length: Desired length of the response ("short", "medium", or "long")
        
    Returns:
        The generated text
    """
    # Create a system message that includes the length control
    system_message = f"""Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant.

IMPORTANT: You must keep your response {length}. Be concise and focused."""
    
    # Create the messages to send to the model
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Generate the response using the Ollama-based LLM
    return llm.invoke(messages)

# Alternative approach using ChatPromptTemplate
def generate_with_template(prompt, length="short"):
    """
    Alternative implementation using ChatPromptTemplate.
    
    Args:
        prompt: The prompt/question to send to the model
        length: Desired length of the response ("short", "medium", or "long")
        
    Returns:
        The generated text
    """
    # Create a template that includes length control
    template = ChatPromptTemplate.from_messages([
        ("system", """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant.

IMPORTANT: You must keep your response {length}. Be concise and focused."""),
        ("human", "{prompt}")
    ])
    
    # Create the chain
    chain = template | llm | StrOutputParser()
    
    # Generate the response
    return chain.invoke({
        "prompt": prompt,
        "length": length
    })

# Example usage
if __name__ == "__main__":
    # Test with the same example as in the original code
    prompt = "Give me a list of wildflowers from Colorado"
    
    print("=== Using direct message approach ===")
    response = generate_with_length_control(prompt, length="short")
    print(response)
    
    print("\n=== Using template approach ===")
    response = generate_with_template(prompt, length="short")
    print(response)
    
    # Compare different length settings
    print("\n=== Comparing different length settings ===")
    lengths = ["short", "medium", "long"]
    
    for length in lengths:
        print(f"\nLength: {length}")
        response = generate_with_template(prompt, length=length)
        print(response)
        
    # Additional example
    print("\n=== Additional example with short length ===")
    historical_prompt = "Explain the significance of the French Revolution"
    response = generate_with_template(historical_prompt, length="short")
    print(response)