from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from typing import Optional

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
)
from src.config.agents import LLMType


def create_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatOpenAI(**llm_kwargs)


def create_deepseek_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["api_base"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


# Cache for LLM instances
_llm_cache: dict[LLMType, ChatOpenAI | ChatDeepSeek] = {}


def get_llm_by_type(llm_type: LLMType) -> ChatOpenAI | ChatDeepSeek:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    if llm_type == "reasoning":
        llm = create_openai_llm(
            model=REASONING_MODEL,
            base_url=REASONING_BASE_URL,
            api_key=REASONING_API_KEY,
        )
    elif llm_type == "basic":
        llm = create_openai_llm(
            model=BASIC_MODEL,
            base_url=BASIC_BASE_URL,
            api_key=BASIC_API_KEY,
        )
    elif llm_type == "vision":
        llm = create_openai_llm(
            model=VL_MODEL,
            base_url=VL_BASE_URL,
            api_key=VL_API_KEY,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    _llm_cache[llm_type] = llm
    return llm


# Initialize LLMs for different purposes - now these will be cached
vl_llm = get_llm_by_type("vision")
'''reasoning_llm = get_llm_by_type("reasoning")
basic_llm = get_llm_by_type("basic")
vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    stream = reasoning_llm.stream("what is mcp?")
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    print(full_response)

    basic_llm.invoke("Hello")
    vl_llm.invoke("Hello")'''
