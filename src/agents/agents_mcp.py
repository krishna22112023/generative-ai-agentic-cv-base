import json
from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.config.tools import MCP_TOOL_MAP
from src.mcp import mcp_client

import sys
import os
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

async def get_react_agent_mcp(agent_name:str):
    
    config_file = os.path.join(root,"src/config/mcp_config.json")
    mcp_client.load_servers(config_path=config_file,tool_names=MCP_TOOL_MAP[agent_name])  
    tools = await mcp_client.start()  
    
    agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP[agent_name]),
    tools=tools,
    prompt=lambda state: apply_prompt_template(agent_name, state),
    )
    return mcp_client,agent

