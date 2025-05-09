from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM  # Updated import
from langchain_core.messages import HumanMessage, SystemMessage

model_name = "granite3.3:8b"  # This should match the name you use with Ollama
ollama_base_url = "http://localhost:11434"  # Default Ollama API URL

# Initialize Ollama LLM with the updated class
llm = OllamaLLM(model=model_name, base_url=ollama_base_url, temperature=0.7, top_p=0.95, repeat_penalty=1.1)

# LangChain Prompt Template for Chat
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),  # System message
    MessagesPlaceholder(variable_name="chat_history"),  # For chat history
    ("human", "{input}")  # Changed "user" to "human" as per LangChain conventions
])

# Create a simple chain
chain = prompt | llm | StrOutputParser()

# Example Usage
output = chain.invoke({"input": "What is the largest ocean on Earth?", "chat_history": []})
print(output)

output = chain.invoke({"input": "Write a short Python function to calculate the area of a circle.", "chat_history": []})
print(output)

# Example of using chat history
chat_history = [
    HumanMessage(content="Hello, who are you?"),
    SystemMessage(content="I am an AI assistant powered by the Granite model through Ollama."),
]

output = chain.invoke({"input": "What can you help me with?", "chat_history": chat_history})
print(output)