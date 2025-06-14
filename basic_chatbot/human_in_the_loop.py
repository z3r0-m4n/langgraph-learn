from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain_core.messages import SystemMessage

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human. Use this when you need expert input or clarification.
    The human's response should be treated as authoritative guidance that you should incorporate into your response."""
    print("\n======= HUMAN ASSISTANCE REQUESTED =======")
    print(f"Query: {query}")
    print("Please provide your expert guidance:")
    human_response = input("> ")
    return f"EXPERT GUIDANCE: {human_response}"

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]

# Create a system message that emphasizes incorporating human feedback
system_message = SystemMessage(content="""You are a helpful AI assistant that can use tools and request human assistance.
When you receive expert guidance from a human:
1. Acknowledge and incorporate their feedback directly
2. Use their guidance to provide more specific and relevant information
3. Ask follow-up questions if you need clarification
4. Build upon their expertise in your responses

Use the web search tool when you need current information or to verify facts.
Use the human assistance tool when you need expert input or clarification.""")

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    # Ensure system message is always first in the conversation
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        state["messages"].insert(0, system_message)
    
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

## user section
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()