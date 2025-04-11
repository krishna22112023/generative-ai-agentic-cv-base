from langgraph.graph import StateGraph, START

from .types import State
from .nodes import (
    coordinator_node,
    supervisor_node,
    data_collection_node,
    data_quality_node,
    data_preprocessor_node,
    data_annotator_node,
    reporter_node,
    planner_node,
)


def build_graph():
    """Build and return the agent workflow graph."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_collector", data_collection_node)
    builder.add_node("data_quality", data_quality_node)
    builder.add_node("data_preprocessor", data_preprocessor_node)
    builder.add_node("data_annotator", data_annotator_node)
    builder.add_node("reporter", reporter_node)
    return builder.compile()
