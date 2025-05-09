from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function

# Define tools as functions with structured input
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # This would normally make an API call to a weather service
    # For this example, we'll return mock data
    return {"temp": 20.5, "unit": "C"}

@tool
def get_stock_price(ticker: str) -> str:
    """Retrieves the current stock price for a given ticker symbol.
    The ticker symbol must be a valid symbol for a publicly traded company
    on a major US stock exchange like NYSE or NASDAQ.
    """
    # This would normally make an API call to a stock API
    # For this example, we'll return mock data
    return f"${150.75}"

# Set up the LLM with Ollama
llm = OllamaLLM(
    model="granite3.3:8b",
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0,
)

# Define the list of tools
tools = [get_current_weather, get_stock_price]

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that has access to tools. Use these tools to answer user questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Format tools to match Ollama's function calling expectations
functions = [format_tool_to_openai_function(t) for t in tools]

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Example usage with chat history
chat_history = []

# Test the agent with a weather query
# user_input = "What's the current weather in New York?"
user_input = "What is the stock price of AAPL?"
try:
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    print("Response:", response["output"])
    
    # Update chat history for potential future queries
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["output"]))
except Exception as e:
    print(f"Error occurred: {e}")
    
    # Try alternative approach if the first one fails
    print("\nAttempting alternative approach...")
    
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    
    # Define tools in the alternative format
    tools_alt = [
        Tool(
            name="get_current_weather",
            func=get_current_weather,
            description="Get the current weather for a given location."
        ),
        Tool(
            name="get_stock_price",
            func=get_stock_price,
            description="Retrieves the current stock price for a given ticker symbol."
        )
    ]
    
    # Initialize agent with a different method
    agent_alt = initialize_agent(
        tools=tools_alt,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Run the alternative agent
    response_alt = agent_alt.run(
        input=user_input,
        chat_history=[]
    )
    
    print("Alternative Response:", response_alt)