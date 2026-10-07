"""LLM configuration.

Uses any OpenAI-compatible endpoint. Defaults to NVIDIA's hosted models;
set LLM_BASE_URL / LLM_MODEL to switch providers (e.g. OpenAI).
"""

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


@lru_cache
def get_llm() -> ChatOpenAI:
    """Create the chat model on first use, so importing the graph needs no API key."""
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY (or OPENAI_API_KEY) in your .env file.")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    # An empty LLM_BASE_URL means "use OpenAI's default endpoint"
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        base_url=base_url or None,
        api_key=api_key,
    )
