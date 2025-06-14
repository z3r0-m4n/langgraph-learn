"""LangGraph agent implementation with a single-node graph.

This module implements a simple agent using LangGraph that processes user input
and generates responses using Claude 3.5 Sonnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent


class Configuration(TypedDict):
    """Configuration parameters for the agent.
    
    These parameters can be set when creating assistants or invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    my_configurable_param: str


@dataclass
class State:
    """State management for the agent.
    
    Attributes:
        input_field: The user's input message
        output: The agent's response (initialized as None)
    """
    input_field: str = "Change me to your input"
    output: str | None = None


def create_agent() -> Any:
    """Creates and configures the agent with the specified model and tools.
    
    Returns:
        A configured agent instance
    """
    return create_react_agent(
        model="anthropic:claude-3-5-sonnet-latest",
        tools=[],
        prompt="You are a helpful assistant. Answer all questions except for those relating to Singapore.",
    )


def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Processes the current state through the model.
    
    Args:
        state: Current state containing the input message
        config: Configuration parameters for the model
        
    Returns:
        Dictionary containing the model's response
    """
    messages = [{"role": "user", "content": state.input_field}]
    
    response = agent.invoke(
        {"messages": messages},
        config
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        output_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        output_content = "No response generated"
    
    return {"output": output_content}


def create_graph() -> StateGraph:
    """Creates and configures the state graph for the agent.
    
    Returns:
        A compiled StateGraph instance
    """
    return (
        StateGraph(State, config_schema=Configuration)
        .add_node("call_model", call_model)
        .add_edge("__start__", "call_model")
        .add_edge("call_model", "__end__")
        .compile()
    )


# Initialize the agent
agent = create_agent()

# Create the graph
graph = create_graph()


def main() -> None:
    """Main entry point for testing the graph."""
    config = {"configurable": {"thread_id": "1"}}
    initial_state = State(input_field="Hello, how are you?")
    
    try:
        result = graph.invoke(initial_state, config)
        print("Input:", result["input_field"])
        print("Output:", result["output"])
    except Exception as e:
        print(f"Error running graph: {e}")


if __name__ == "__main__":
    main()