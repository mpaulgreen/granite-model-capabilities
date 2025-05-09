from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Update to your specific Granite model
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.1,  # Lower temperature for summarization tasks
)

# Create a system prompt that mimics Granite's system prompt
system_prompt = """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant."""

# Create the prompt template for summarization
summarization_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Summarize a fragment of an interview transcript. In this interview, an NBC reporter interviews Simone Biles about her participation in Paris 2024 Olympic games.
Your response should only include the answer. Do not provide any further explanation.
{transcript}""")
])

# Create the chain
summarization_chain = summarization_template | llm | StrOutputParser()

def summarize_interview(transcript: str) -> str:
    """
    Summarize an interview transcript using the Granite model.
    
    Args:
        transcript: The interview transcript to summarize
        
    Returns:
        A summary of the interview transcript
    """
    return summarization_chain.invoke({"transcript": transcript})

# Example usage
if __name__ == "__main__":
    # The sample transcript from the user's example
    sample_transcript = """Speaker 1 (00:00):
Simone, congratulations.
Simone (00:02):
Thank you."""
    
    # Get and print the summary
    summary = summarize_interview(sample_transcript)
    print("Summary:")
    print(summary)
    
    # You can also try with a longer transcript
    longer_transcript = """Speaker 1 (00:00):
Simone, congratulations.
Simone (00:02):
Thank you.
Speaker 1 (00:04):
How does it feel to add another gold medal to your collection?
Simone (00:08):
It's surreal. After everything that happened in Tokyo, to come back and perform the way I did, I'm just really proud of myself and the team.
Speaker 1 (00:15):
Can you tell us about the preparation for these Olympics?
Simone (00:18):
It was different this time. I focused a lot on mental health and finding joy in the sport again. My coaches were amazing in helping me rebuild confidence in my abilities."""
    
    print("\nLonger Transcript Summary:")
    print(summarize_interview(longer_transcript))