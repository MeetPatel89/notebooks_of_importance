import os
from typing import List, Optional

from openai import OpenAI

from ml_boilerplate_module.llm.interfaces import LLMClient, LLMResponse, Message
from ml_boilerplate_module.llm.message import to_openai_message


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        print(f"Using OpenAI model: {self.model}")
        self.openai = OpenAI()

    def send_message(
        self,
        messages: List[Message],
        system_message: Optional[str] = None,
    ) -> LLMResponse:
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=to_openai_message(messages, system_message),
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            provider="openai",
            model=self.model,
            raw=response,
        )
