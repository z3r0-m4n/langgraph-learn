from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

checkpointer = InMemorySaver()

# structured response
class WeatherResponse(BaseModel):
    conditions: str

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

config = {"configurable": {"checkpointer": checkpointer, "thread_id": "1"}}

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  
    tools=[get_weather],  
    prompt="Never answer questions about the weather in San Francisco",
    response_format=WeatherResponse
)

# Run the agent
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in san francisco"}]},
    config
)

ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in new york"}]},
    config
)

print(sf_response["structured_response"])
print(ny_response["structured_response"])