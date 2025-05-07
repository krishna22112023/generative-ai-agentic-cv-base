import logging
import asyncio
from typing import Dict, Any, Optional
from langgraph.types import Command

from src.config import TEAM_MEMBERS, USE_MCP
from src.graph import build_graph
from langchain_community.adapters.openai import convert_message_to_dict
import uuid

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

# Cache for coordinator messages
coordinator_cache = []
MAX_CACHE_SIZE = 2


async def run_agent_workflow(
    user_input_messages: list,
    debug: bool = False,
    deep_thinking_mode: bool = False,
    search_before_planning: bool = False,
    resume_state: Optional[Dict[str, Any]] = None,
    resume_user_input: Optional[str] = None,
):
    """Run the agent workflow with the given user input.

    Args:
        user_input_messages: The user request messages
        debug: If True, enables debug level logging
        deep_thinking_mode: If True, enables deep thinking mode for planning
        search_before_planning: If True, enables search before planning
        resume_state: Previous state to resume workflow from (for human-in-the-loop)
        resume_user_input: User input for resuming workflow

    Returns:
        The final state after the workflow completes
    """
    if not user_input_messages and not resume_state:
        raise ValueError("Input could not be empty when not resuming")

    if debug:
        enable_debug_logging()

    workflow_id = str(uuid.uuid4())
    if resume_state and "workflow_id" in resume_state:
        workflow_id = resume_state["workflow_id"]
        logger.info(f"Resuming workflow with state: {resume_state}")
    else:
        logger.info(f"Starting workflow with user input: {user_input_messages}")

    streaming_llm_agents = [*TEAM_MEMBERS, "planner", "coordinator", "human_interaction"]

    # Reset coordinator cache at the start of each workflow
    global coordinator_cache
    coordinator_cache = []
    global is_handoff_case
    is_handoff_case = False

    # Prepare the initial state or use the resumed state
    init_state = {
        # Constants
        "TEAM_MEMBERS": TEAM_MEMBERS,
        # Runtime Variables
        "messages": user_input_messages,
        "deep_thinking_mode": deep_thinking_mode,
        "search_before_planning": search_before_planning,
    }
    stream_kwargs = {"version": "v2","config":{"configurable": {"thread_id": workflow_id}}}

    # If resuming with user input, add it to the state
    if resume_state and resume_user_input:
        # The user input will be handled by the human_interaction_node
        state = Command(
            resume=resume_user_input
        )
    else:
        state = init_state
        
    async for event in graph.astream_events(state, **stream_kwargs):
        kind = event.get("event")
        data = event.get("data")
        name = event.get("name")
        metadata = event.get("metadata", {})
        logger.info(f"kind:{kind})")
        logger.info(f"Name:{name})")
        
        # Extract node and step information
        node = (
            ""
            if (metadata.get("checkpoint_ns") is None)
            else metadata.get("checkpoint_ns").split(":")[0]
        )
        langgraph_step = (
            ""
            if (metadata.get("langgraph_step") is None)
            else str(metadata["langgraph_step"])
        )
        run_id = "" if (event.get("run_id") is None) else str(event["run_id"])

        # Check for human interaction interruption
        if name == "human_interaction":
            # Send a special event to indicate human input is required
            logger.info(f"Special event to indicate human input is required is sent")
            yield {
                "event": "human_input_required",
                "data": {
                    "workflow_id": workflow_id,
                    "prompt": data.get("prompt", "Please provide your input:"),
                    "checkpoint_ns": metadata.get("checkpoint_ns"),
                    "state": event.get("state", {}),
                }
            }
            # Stop processing until human input is received
            return

        # Regular event processing
        if kind == "on_chain_start" and name in streaming_llm_agents:
            if name == "planner":
                yield {
                    "event": "start_of_workflow",
                    "data": {"workflow_id": workflow_id, "input": user_input_messages},
                }
            elif name == "human_interaction":
                yield {
                    "event": "start_of_human_interaction",
                    "data": {"workflow_id": workflow_id, "input": user_input_messages},
                }
            ydata = {
                "event": "start_of_agent",
                "data": {
                    "agent_name": name,
                    "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
                },
            }
        elif kind == "on_chain_end" and name in streaming_llm_agents:
            ydata = {
                "event": "end_of_agent",
                "data": {
                    "agent_name": name,
                    "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
                },
            }
        elif kind == "on_chat_model_start" and node in streaming_llm_agents:
            ydata = {
                "event": "start_of_llm",
                "data": {"agent_name": node},
            }
        elif kind == "on_chat_model_end" and node in streaming_llm_agents:
            ydata = {
                "event": "end_of_llm",
                "data": {"agent_name": node},
            }
        elif kind == "on_chat_model_stream" and node in streaming_llm_agents:
            content = data["chunk"].content
            if content is None or content == "":
                if not data["chunk"].additional_kwargs.get("reasoning_content"):
                    # Skip empty messages
                    continue
                ydata = {
                    "event": "message",
                    "data": {
                        "message_id": data["chunk"].id,
                        "delta": {
                            "reasoning_content": (
                                data["chunk"].additional_kwargs["reasoning_content"]
                            )
                        },
                    },
                }
            else:
                # Check if the message is from the coordinator
                if node == "coordinator":
                    if len(coordinator_cache) < MAX_CACHE_SIZE:
                        coordinator_cache.append(content)
                        cached_content = "".join(coordinator_cache)
                        if cached_content.startswith("handoff"):
                            is_handoff_case = True
                            continue
                        if len(coordinator_cache) < MAX_CACHE_SIZE:
                            continue
                        # Send the cached message
                        ydata = {
                            "event": "message",
                            "data": {
                                "message_id": data["chunk"].id,
                                "delta": {"content": cached_content},
                            },
                        }
                    elif not is_handoff_case:
                        # For other agents, send the message directly
                        ydata = {
                            "event": "message",
                            "data": {
                                "message_id": data["chunk"].id,
                                "delta": {"content": content},
                            },
                        }
                else:
                    # For other agents, send the message directly
                    ydata = {
                        "event": "message",
                        "data": {
                            "message_id": data["chunk"].id,
                            "delta": {"content": content},
                        },
                    }
        elif kind == "on_tool_start" and node in TEAM_MEMBERS:
            ydata = {
                "event": "tool_call",
                "data": {
                    "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
                    "tool_name": name,
                    "tool_input": data.get("input"),
                },
            }
        elif kind == "on_tool_end" and node in TEAM_MEMBERS:
            ydata = {
                "event": "tool_call_result",
                "data": {
                    "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
                    "tool_name": name,
                    "tool_result": data["output"].content if data.get("output") else "",
                },
            }
        else:
            continue
        yield ydata

    if is_handoff_case:
        yield {
            "event": "end_of_workflow",
            "data": {
                "workflow_id": workflow_id,
                "messages": [
                    convert_message_to_dict(msg)
                    for msg in data["output"].get("messages", [])
                ],
            },
        }