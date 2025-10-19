"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState
from shared.utils import format_docs, load_chat_model

async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.messages[-1].content})
    return {"documents": result["documents"]}


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(documents=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    print("responsd.messaages", messages)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(conduct_research)
builder.add_node(respond)

builder.add_edge(START, "conduct_research")
builder.add_edge("conduct_research", "respond")
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
