from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
import random

# Set seed for reproducibility (similar to set_seed in transformers)
random.seed(42)

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Make sure this matches your Ollama model name
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.7,
    max_tokens=8192,  # Equivalent to max_new_tokens in the original code
    # Additional parameters to pass to Ollama
    stop=None,
    seed=42,  # Set seed for reproducibility
)

def enable_thinking_mode(prompt):
    """
    Enable the "thinking" mode for Granite model via Ollama.
    
    In the original transformers code, this is done with thinking=True parameter.
    Here we'll implement it through a modified system prompt.
    
    Args:
        prompt: The prompt/question to send to the model
        
    Returns:
        The generated text with thinking process included
    """
    # Create a system message that encourages thinking step by step
    system_message = """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant.

When answering questions, especially math or reasoning problems, please show your thinking step by step 
before giving the final answer. Walk through your reasoning process thoroughly, considering different aspects
of the problem and explaining your approach. This will help ensure accuracy and show how you arrived at the solution."""
    
    # Create the messages to send to the model
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Generate the response using the Ollama-based LLM
    return llm.invoke(messages)

# Alternative approach using ChatPromptTemplate
def thinking_with_template(prompt):
    """
    Alternative implementation using ChatPromptTemplate.
    
    Args:
        prompt: The prompt/question to send to the model
        
    Returns:
        The generated text with thinking process
    """
    # Create a template that encourages thinking
    template = ChatPromptTemplate.from_messages([
        ("system", """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant.

When answering questions, especially math or reasoning problems, please show your thinking step by step 
before giving the final answer. Walk through your reasoning process thoroughly, considering different aspects
of the problem and explaining your approach. This will help ensure accuracy and show how you arrived at the solution."""),
        ("human", "{prompt}")
    ])
    
    # Create the chain
    chain = template | llm | StrOutputParser()
    
    # Generate the response
    return chain.invoke({"prompt": prompt})

# Example usage
if __name__ == "__main__":
    # Test with the same example as in the original code
    prompt = "A farmer has 10 cows, 5 chickens, and 2 horses. If we count all the animals' legs together, how many legs are there in total?"
    
    print("=== Using direct message approach with thinking mode ===")
    response = enable_thinking_mode(prompt)
    print(response)
    
    print("\n=== Using template approach with thinking mode ===")
    response = thinking_with_template(prompt)
    print(response)
    
    # Additional examples to test thinking mode
    additional_examples = [
        "If a train travels at 60 mph and needs to cover 240 miles, how long will the journey take?",
        "A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?",
        "A car costs $20,000 and depreciates by 15% each year. What will be its value after 3 years?"
    ]
    
    print("\n=== Additional examples with thinking mode ===")
    for i, example in enumerate(additional_examples, 1):
        print(f"\nExample {i}: {example}")
        response = thinking_with_template(example)
        print(response)