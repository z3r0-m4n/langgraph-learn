from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage

# Define our state type
class AgentState(TypedDict):
    messages: List[BaseMessage]

# Create our tools
tool = TavilySearch(max_results=2)
tools = [tool]

# Create the agent
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=tools,
    prompt='''You are a helpful assistant. You have access to a web search tool that you should use when:
    1. You need current information
    2. You're asked about recent events
    3. You need to verify facts
    4. You're unsure about something
    Always use the web search tool when appropriate, and explain to the user when you're searching for information.'''
)

def print_ai_message(state: AgentState) -> None:
    """Print the AI's response from the state."""
    print("=======AI RESPONSE=======")
    print(state["messages"][-1].content)

def should_continue(state: AgentState) -> str:
    """Determine if we should continue the conversation.
    For now, we'll end after the agent's first response."""
    # If we have at least one AI message, end the conversation
    if any(isinstance(msg, AIMessage) for msg in state["messages"]):
        return "end"
    return "continue"

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("print_response", print_ai_message)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "print_response")
workflow.add_conditional_edges(
    "print_response",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# Compile the graph
app = workflow.compile()

# Run the graph
if __name__ == "__main__":
    # Initial state
    state = {"messages": [{"role": "user", "content": "what is the weather in singapore?"}]}
    
    # Run the graph
    for output in app.stream(state):
        # The graph will automatically print responses through the print_response node
        pass