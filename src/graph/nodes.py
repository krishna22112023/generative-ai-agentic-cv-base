import logging
import json
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command,interrupt

from src.agents import data_collector_agent, data_quality_agent, data_preprocessor_agent, data_annotator_agent
from src.agents import get_react_agent_mcp
from src.agents.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from .types import State, Router

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


def data_collection_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data collection agent that imports data from external data sources, such as minIO"""
    logger.info("Data Collection agent starting task")
    result = data_collector_agent.invoke(state)
    logger.info("Data Collection agent completed task")
    logger.debug(f"Data Collection agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_collector", result["messages"][-1].content
                    ),
                    name="data_collector",
                )
            ]
        },
        goto="supervisor",
    )

async def data_collection_anode(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data collection agent with mcp connection to data sources"""
    logger.info("Data Collection agent starting task")
    client,data_collector_agent = await get_react_agent_mcp("data_collector")
    result = await data_collector_agent.ainvoke(state)
    logger.info("Data Collection agent completed task")
    logger.debug(f"Data Collection agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_collector", result["messages"][-1].content
                    ),
                    name="data_collector",
                )
            ]
        },
        goto="supervisor",
    )



def data_quality_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data quality agent that assess quality of image data from external data sources."""
    logger.info("data quality agent starting task")
    result = data_quality_agent.invoke(state)
    logger.info("data quality agent completed task")
    logger.debug(f"data quality agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_quality", result["messages"][-1].content
                    ),
                    name="data_quality",
                )
            ]
        },
        goto="supervisor",
    )

async def data_quality_anode(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data quality agent that assess quality of image data from external data sources."""
    logger.info("data quality agent starting task")
    client,data_quality_agent = await get_react_agent_mcp("data_quality")
    result = await data_quality_agent.ainvoke(state)
    logger.info("data quality agent completed task")
    logger.debug(f"data quality agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_quality", result["messages"][-1].content
                    ),
                    name="data_quality",
                )
            ]
        },
        goto="supervisor",
    )

def data_preprocessor_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data preprocessor agent that performs image restoration based on the quality assessment"""
    logger.info("data preprocessor agent starting task")
    result = data_preprocessor_agent.invoke(state)
    logger.info("data preprocessor agent completed task")
    logger.debug(f"data preprocessor agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_preprocessor", result["messages"][-1].content
                    ),
                    name="data_preprocessor",
                )
            ]
        },
        goto="supervisor",
    )

async def data_preprocessor_anode(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data preprocessor agent that performs image restoration based on the quality assessment"""
    logger.info("data preprocessor agent starting task")
    client,data_preprocessor_agent = await get_react_agent_mcp("data_preprocessor")
    result = await data_preprocessor_agent.ainvoke(state)
    logger.info("data preprocessor agent completed task")
    logger.debug(f"data preprocessor agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_preprocessor", result["messages"][-1].content
                    ),
                    name="data_preprocessor",
                )
            ]
        },
        goto="supervisor",
    )


def data_annotator_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data annotator agent that performs annotation on the processed data"""
    logger.info("data annotator agent starting task")
    result = data_annotator_agent.invoke(state)
    logger.info("data annotator agent completed task")
    logger.debug(f"data annotator agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_annotator", result["messages"][-1].content
                    ),
                    name="data_annotator",
                )
            ]
        },
        goto="supervisor",
    )

async def data_annotator_anode(state: State) -> Command[Literal["supervisor"]]:
    """Node for the data annotator agent that performs annotation on the processed data"""
    logger.info("data annotator agent starting task")
    client,data_preprocessor_agent = await get_react_agent_mcp("data_annotator")
    result = await data_preprocessor_agent.ainvoke(state)
    logger.info("data annotator agent completed task")
    logger.debug(f"data annotator agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "data_annotator", result["messages"][-1].content
                    ),
                    name="data_annotator",
                )
            ]
        },
        goto="supervisor",
    )

def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"coordinator response: {response}")

    goto = "__end__"
    if "handoff_to_planner" in response.content:
        goto = "planner"

    return Command(
        goto=goto,
    )

def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    messages = apply_prompt_template("supervisor", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        .with_structured_output(Router)
        .invoke(messages)
    )
    goto = response["next"]
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Supervisor response: {response}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    return Command(goto=goto, update={"next": goto})


def planner_node(state: State) -> Command[Literal["human_interaction", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    messages = apply_prompt_template("planner", state)
    # whether to enable deep thinking mode
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    stream = llm.stream(messages)
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "human_interaction"
    try:
        json.loads(full_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )

def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {response}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format("reporter", response.content),
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )

def human_interaction_node(state: State) -> Command[Literal["planner", "supervisor"]]:
    """Human in the loop node that ask for human feedback."""
    logger.info("Human in the loop active")
    prompt = apply_prompt_template("human_interaction", state)
    prompt_text = prompt[0]["content"]
    logger.info(f"state in human interaction node {state}")

    # If resuming with user input, get it from state
    if state.get("human_in_the_loop"):
        user_response = state["human_in_the_loop"]
        logger.info(f"Resuming with user input: {user_response}")
        # Explicitly clear the flag after consuming it
        update_dict = {
            "messages": [
                HumanMessage(
                    content=f"Human feedback: {user_response}",
                    name="human_interaction",
                )
            ],
            # Remove the key from state by deleting it if present
        }
        # Remove the key from the state dict if present (in-place)
        if "human_in_the_loop" in state:
            del state["human_in_the_loop"]
    else:
        # Interrupt and wait for user input
        user_response = interrupt(prompt_text)
        logger.info(f"User input: {user_response}")
        update_dict = {
            "messages": [
                HumanMessage(
                    content=f"Human feedback: {user_response}",
                    name="human_interaction",
                )
            ],
            "human_in_the_loop": user_response
        }

    logger.info(f"routing based on user input: {user_response}")
    next_node = "supervisor" if "approve" in user_response.lower() else "planner"
    logger.info(f"Next node based on user input: {next_node}")
    # Resume graph execution with the user's input
    return Command(
        update=update_dict,
        goto=next_node
    )