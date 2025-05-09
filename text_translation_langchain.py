from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Update to your specific Granite model
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.2,  # Lower temperature for accurate translations
    stop=["<|end_of_text|>"]  # Stop sequence to match Granite's output format
)

# Create a system prompt that mimics Granite's system prompt
system_prompt = """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant."""

# Create the prompt template for translation
translation_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Please translate the following {source_language} text to {target_language}. {additional_instructions}

{text}

Please only output the translation, and nothing else.""")
])

# Create the chain
translation_chain = translation_template | llm | StrOutputParser()

def translate_text(text, source_language="English", target_language="Spanish", additional_instructions=""):
    """
    Translate text from source language to target language.
    
    Args:
        text: The text to translate
        source_language: The source language (default: English)
        target_language: The target language (default: Spanish)
        additional_instructions: Any additional instructions for the translation
        
    Returns:
        The translated text
    """
    return translation_chain.invoke({
        "text": text,
        "source_language": source_language,
        "target_language": target_language,
        "additional_instructions": additional_instructions
    }).strip()

# Example usage
if __name__ == "__main__":
    # Example from your prompt
    english_greetings = "Morning!, how are things?, hello, it's good to see you, what's up?"
    
    # Get and print the translation
    spanish_translation = translate_text(english_greetings)
    print(f"English: {english_greetings}")
    print(f"Spanish: {spanish_translation}")
    
    # Additional examples with different language pairs
    print("\nAdditional Examples:")
    
    # English to French
    english_text = "I would like to order a coffee and a croissant, please."
    french_translation = translate_text(
        text=english_text,
        target_language="French"
    )
    print(f"\nEnglish: {english_text}")
    print(f"French: {french_translation}")
    
    # English to German with specific instructions
    business_text = "We look forward to our meeting next week to discuss the project timeline."
    german_translation = translate_text(
        text=business_text,
        target_language="German",
        additional_instructions="Use formal business language."
    )
    print(f"\nEnglish: {business_text}")
    print(f"German: {german_translation}")
    
    # Spanish to English
    spanish_text = "Necesito encontrar la estación de tren más cercana. ¿Puedes ayudarme?"
    english_translation = translate_text(
        text=spanish_text,
        source_language="Spanish",
        target_language="English"
    )
    print(f"\nSpanish: {spanish_text}")
    print(f"English: {english_translation}")