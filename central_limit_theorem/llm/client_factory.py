from typing import Any

from ml_boilerplate_module.llm.anthropic_client import AnthropicAIClient
from ml_boilerplate_module.llm.google_client import GoogleAIClient
from ml_boilerplate_module.llm.grok_client import GrokAIClient
from ml_boilerplate_module.llm.groq_client import GroqAIClient
from ml_boilerplate_module.llm.interfaces import LLMClient
from ml_boilerplate_module.llm.openai_client import OpenAIClient


def get_llm_client(provider: str, **kwargs: Any) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicAIClient(**kwargs)
    elif provider == "google":
        return GoogleAIClient(**kwargs)
    elif provider == "groq":
        return GroqAIClient(**kwargs)
    elif provider == "grok":
        return GrokAIClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
