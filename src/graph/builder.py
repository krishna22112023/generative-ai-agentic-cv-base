from langgraph.graph import StateGraph, START
#from langgraph.checkpoint.memory import InMemorySaver
from .types import State

def build_graph(use_mcp:bool=True):
    """Build and return the agent workflow graph.
    If use_mcp is enabled, then interface tools on mcp server"""

    if use_mcp:
        from .nodes import (
        coordinator_node,
        supervisor_node,
        data_collection_anode,
        data_quality_anode,
        data_preprocessor_anode,
        data_annotator_anode,
        reporter_node,
        planner_node
        )
        builder = StateGraph(State)
        builder.add_edge(START, "coordinator")
        builder.add_node("coordinator", coordinator_node)
        builder.add_node("planner", planner_node)
        #builder.add_node("human_interaction",human_interaction_node)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("data_collector", data_collection_anode)
        builder.add_node("data_quality", data_quality_anode)
        builder.add_node("data_preprocessor", data_preprocessor_anode)
        builder.add_node("data_annotator", data_annotator_anode)
        builder.add_node("reporter", reporter_node)
    else:
        from .nodes import (
        coordinator_node,
        supervisor_node,
        data_collection_node,
        data_quality_node,
        data_preprocessor_node,
        data_annotator_node,
        reporter_node,
        planner_node
        )
        builder = StateGraph(State)
        builder.add_edge(START, "coordinator")
        builder.add_node("coordinator", coordinator_node)
        builder.add_node("planner", planner_node)
        #builder.add_node("human_interaction",human_interaction_node)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("data_collector", data_collection_node)
        builder.add_node("data_quality", data_quality_node)
        builder.add_node("data_preprocessor", data_preprocessor_node)
        builder.add_node("data_annotator", data_annotator_node)
        builder.add_node("reporter", reporter_node)
    #checkpointer = InMemorySaver()
    return builder.compile()
