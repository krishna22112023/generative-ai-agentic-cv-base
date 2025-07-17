from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    no_reference_iqa,
    verify_no_reference_iqa,
    preprocessing_pipeline,
    grounded_annotator,
    tavily_tool,
    browser_tool
)

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

# Create agents using configured LLM types

data_collector_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_collector"]),
    tools=[tavily_tool,browser_tool],
    prompt=lambda state: apply_prompt_template("data_collector", state),
)

data_quality_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_quality"]),
    tools=[no_reference_iqa],
    prompt=lambda state: apply_prompt_template("data_quality", state),
)

data_preprocessor_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_preprocessor"]),
    tools=[preprocessing_pipeline,verify_no_reference_iqa],
    prompt=lambda state: apply_prompt_template("data_preprocessor", state),
)

data_annotator_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_annotator"]),
    tools=[grounded_annotator],
    prompt=lambda state: apply_prompt_template("data_annotator", state),
)



