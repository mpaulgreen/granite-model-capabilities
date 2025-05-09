from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Update to your specific Granite model
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.1,  # Lower temperature for classification tasks
    top_p=0.95,      # Tight sampling for classification
)

# Create a system prompt that mimics Granite's system prompt
system_prompt = """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant."""

# Examples for few-shot learning to help the model understand the task
examples = """
Review: Oh, where do I even begin? Barbie 2023 is a tour de force that has left me utterly captivated, enchanted, and spellbound. Every moment of this cinematic marvel was nothing short of pure excellence, deserving nothing less than a perfect 10 out of 10 rating!
Sentiment: positive

Review: It's a shame because I love Little Women and Lady Bird. Barbie did not serve. It's a long ramble about how matriarchy is perfect and patriarchy is stupid. I agree with the latter, but the execution is just awful. Mattel's CEO and corporate people started as a mockery of men having all the top positions in a company; as the movie goes, they're just there at the back of your mind, and it's an unpleasant experience because they did nothing after. There were so many unnecessary scenes that could have been meaningful. For example, what's the scene where Allan fights other Kens for? I'm also surprised that there were no gay Barbies and Kens. And I thought this was supposed to "break" gender norms. So many seeds were planted at once, but none reached their full potential.
Sentiment: negative
"""

# Create the prompt template for sentiment classification
classification_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Classify the sentiment of the movie reviews as positive or negative. Your response should only include the answer. Do not provide any further explanation. Here are some examples, complete the last one:

{examples}

Review: {review}
Sentiment:""")
])

# Create the chain
classification_chain = classification_template | llm | StrOutputParser()

def classify_sentiment(review: str) -> str:
    """
    Classify the sentiment of a movie review as positive or negative.
    
    Args:
        review: The movie review text to classify
        
    Returns:
        The sentiment classification: "positive" or "negative"
    """
    return classification_chain.invoke({
        "examples": examples,
        "review": review
    }).strip()

# Example usage
if __name__ == "__main__":
    # The sample reviews
    test_review = """Russell Crowe's new action movie Land of Bad can't break the star's recent streak of poor Rotten Tomatoes ratings. Co-starring Liam and Luke Hemsworth, Crowe's latest action effort concerns a special ops mission in the Philippines that goes wrong, leading to a harrowing fight for survival. Crowe plays Reaper, a drone pilot who must guide a group of outnumbered and stranded soldiers back to safety. Despite the Land of Bad cast including big names like Crowe and two of the Hemsworths, the action film has failed to impress critics, currently sitting at 52% fresh on Rotten Tomatoes on 24 reviews, becoming the fifth straight Crowe-starring film to be certified rotten by the review aggregator."""
    
    # Get and print the sentiment classification
    sentiment = classify_sentiment(test_review)
    print(f"Review Sentiment: {sentiment}")
    
    # Additional examples to test the model
    additional_reviews = [
        "I absolutely loved this movie! The acting was superb and the storyline kept me engaged throughout. Highly recommend!",
        "What a waste of time and money. The plot made no sense, the characters were one-dimensional, and I kept checking my watch.",
        "While not perfect, the film offers some genuinely moving moments and solid performances from the main cast."
    ]
    
    print("\nAdditional Reviews Classifications:")
    for i, review in enumerate(additional_reviews, 1):
        sentiment = classify_sentiment(review)
        print(f"{i}. Sentiment: {sentiment}")
        print(f"   Review snippet: {review[:50]}...")
        print()