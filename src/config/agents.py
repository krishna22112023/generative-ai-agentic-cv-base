from typing import Literal

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "supervisor": "basic", 
    "data_collector": "basic",  
    "data_quality": "basic",  
    "data_preprocessor": "basic",  
    "data_annotator": "basic",  
}

