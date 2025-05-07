import logging
from src.config import TEAM_MEMBERS,USE_MCP
from src.graph import build_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph(use_mcp=USE_MCP)


def run_agent_workflow(user_input: str, debug: bool = False):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input}")
    if USE_MCP:
        result = graph.ainvoke(
            {
                # Constants
                "TEAM_MEMBERS": TEAM_MEMBERS,
                # Runtime Variables
                "messages": [{"role": "user", "content": user_input}],
                "deep_thinking_mode": False,
                "search_before_planning": True,
            }
        )
    else:
        result = graph.invoke(
            {
                # Constants
                "TEAM_MEMBERS": TEAM_MEMBERS,
                # Runtime Variables
                "messages": [{"role": "user", "content": user_input}],
                "deep_thinking_mode": False,
                "search_before_planning": True,
            }
        )
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
    with open("./assets/langgraph_diagram.mmd", "w") as f:
        f.write(graph.get_graph().draw_mermaid())
