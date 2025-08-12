from typing import List, Optional

from anthropic import Anthropic

from ml_boilerplate_module.llm.interfaces import LLMClient, LLMResponse, Message
from ml_boilerplate_module.llm.message import to_anthropic_message


class AnthropicAIClient(LLMClient):
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        self.model = model
        print(f"Using Anthropic model: {self.model}")
        self.anthropic = Anthropic()

    def send_message(
        self,
        messages: List[Message],
        system_message: Optional[str] = None,
    ) -> LLMResponse:
        response = self.anthropic.messages.create(
            model=self.model,
            system=system_message or "You are a helpful assistant.",
            messages=to_anthropic_message(messages),
            max_tokens=4000,
        )
        return LLMResponse(
            content=response.content[0].text
            if hasattr(response.content[0], "text")
            else str(response.content[0]),
            provider="anthropic",
            model=self.model,
            raw=response,
        )
