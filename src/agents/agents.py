from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    list_objects,
    download_objects,
    upload_objects,
    delete_objects,
    openai_vlm_iqa,
    create_ir_pipeline,
    run_ir_pipeline
)

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

# Create agents using configured LLM types

data_collector_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_collector"]),
    tools=[list_objects,download_objects,upload_objects,delete_objects],
    prompt=lambda state: apply_prompt_template("data_collector", state),
)

data_quality_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_quality"]),
    tools=[openai_vlm_iqa],
    prompt=lambda state: apply_prompt_template("data_quality", state),
)

data_preprocessor_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["data_preprocessor"]),
    tools=[create_ir_pipeline,run_ir_pipeline],
    prompt=lambda state: apply_prompt_template("data_preprocessor", state),
)



