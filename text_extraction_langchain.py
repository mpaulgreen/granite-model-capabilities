from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any

# Initialize the Ollama LLM with Granite model
llm = OllamaLLM(
    model="granite3.3:8b",  # Update to your specific Granite model
    base_url="http://localhost:11434",  # Default Ollama API URL
    temperature=0.1,  # Lower temperature for precise extraction tasks
    top_p=0.95,      # Tight sampling for factual extraction
    stop=["<|end_of_text|>"]  # Stop sequence to match Granite's output format
)

# Create a system prompt that mimics Granite's system prompt
system_prompt = """Knowledge Cutoff Date: April 2024.
Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful AI assistant."""

# Examples for few-shot learning to help the model understand the extraction task
examples = """
10K Sentence: The credit agreement also provides that up to $500 million in commitments may be used for letters of credit.
Line Of Credit Facility Maximum Borrowing Capacity: $500M

10K Sentence: In March 2020, we upsized the Credit Agreement by $100 million, which matures July 2023, to $2.525 billion.
Line Of Credit Facility Maximum Borrowing Capacity: $2.525B
"""

# Create the prompt template for extraction
extraction_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Extract the Line Of Credit Facility Maximum Borrowing Capacity from the 10K sentences. Your response should only include the answer. Do not provide any further explanation. Here are some examples, complete the last one:

{examples}

10K Sentence: {text}
Line Of Credit Facility Maximum Borrowing Capacity:""")
])

# Create the chain
extraction_chain = extraction_template | llm | StrOutputParser()

def extract_credit_facility_amount(text: str) -> str:
    """
    Extract the Line Of Credit Facility Maximum Borrowing Capacity from 10K sentences.
    
    Args:
        text: The 10K sentence text to analyze
        
    Returns:
        The extracted credit facility amount
    """
    return extraction_chain.invoke({
        "examples": examples,
        "text": text
    }).strip()

# Example usage
if __name__ == "__main__":
    # The sample 10K text from the example
    test_text = """We prepared our impairment test as of October 1, 2022 and determined that the fair values of each of our reporting units exceeded net book value by more than 50%. Among our reporting units, the narrowest difference between the calculated fair value and net book value was in our Principal Markets segment's Canada reporting unit, whose calculated fair value exceeded its net book value by 53%. Future developments related to macroeconomic factors, including increases to the discount rate used, or changes to other inputs and assumptions, including revenue growth, could reduce the fair value of this and/or other reporting units and lead to impairment. There were no goodwill impairment losses recorded for the nine months ended December 31, 2022. Cumulatively, the Company has recorded $469 million in goodwill impairment charges within its former EMEA ($293 million) and current United States ($176 million) reporting units. Revolving Credit Agreement In October 2021, we entered into a $3.15 billion multi-currency revolving credit agreement (the "Revolving Credit Agreement") for our future liquidity needs. The Revolving Credit Agreement expires, unless extended, in October 2026. Interest rates on borrowings under the Revolving Credit Agreement are based on prevailing market interest rates, plus a margin, as further described in the Revolving Credit Agreement. The total expense recorded by the Company for the Revolving Credit Agreement was not material in any of the periods presented. We may voluntarily prepay borrowings under the Revolving Credit Agreement without premium or penalty, subject to customary "breakage" costs. The Revolving Credit Agreement includes certain customary mandatory prepayment provisions. Interest on Debt Interest expense for the three and nine months ended December 31, 2022 was $27 million and $65 million, compared to $18 million and $50 million for the three and nine months ended December 31, 2021. Most of the interest for the pre-Separation period presented in the historical Consolidated Income Statement reflects the allocation of interest expense associated with debt issued by IBM from which a portion of the proceeds benefited Kyndryl."""
    
    # Get and print the extraction
    extracted_amount = extract_credit_facility_amount(test_text)
    print(f"Line Of Credit Facility Maximum Borrowing Capacity: {extracted_amount}")
    
    # Additional examples to test the model
    additional_texts = [
        "As of December 31, 2023, we had $2.1 billion in outstanding borrowings under our revolving Credit Facility, which has a maximum borrowing capacity of $4.5 billion.",
        "On March 15, 2023, the Company entered into a new credit agreement providing for a 5-year senior unsecured revolving credit facility in the amount of $750 million.",
        "Under the terms of the amended Revolving Credit Facility, the maximum borrowing capacity was increased from $1.75 billion to $2 billion, and the maturity date was extended to April 2025."
    ]
    
    print("\nAdditional Examples:")
    for i, text in enumerate(additional_texts, 1):
        extracted = extract_credit_facility_amount(text)
        print(f"\nExample {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Line Of Credit Facility Maximum Borrowing Capacity: {extracted}")