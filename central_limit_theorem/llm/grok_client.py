from typing import List, Optional

from xai_sdk import Client as GrokClient  # type: ignore
from xai_sdk.chat import system, user  # type: ignore

from ml_boilerplate_module.llm.interfaces import LLMClient, LLMResponse, Message


class GrokAIClient(LLMClient):
    def __init__(self, model: str = "grok-4"):
        self.model = model
        print(f"Using Grok model: {self.model}")
        self.grok = GrokClient()

    def send_message(
        self,
        messages: List[Message],
        system_message: Optional[str] = None,
    ) -> LLMResponse:
        user_messages = [user(msg.content) for msg in messages if msg.role == "user"]
        response = self.grok.chat.create(
            model=self.model,
            messages=[system(system_message), *user_messages],
        )
        print("Response Type: ", type(response))
        return LLMResponse(
            content=response.sample().content,
            provider="grok",
            model=self.model,
            raw=response,
        )
