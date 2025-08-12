from typing import Any, Dict, Iterator, List, Optional

from anthropic import MessageStreamManager
from groq import Groq
from groq._types import NOT_GIVEN, NotGiven
from groq.types.chat import ChatCompletionToolParam
from openai._streaming import Stream
from openai.types.chat import ChatCompletionChunk

from ml_boilerplate_module.llm.interfaces import LLMClient
from ml_boilerplate_module.llm.message import to_openai_message


class GroqAIClient(LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        print(f"Using Groq model: {self.model}")
        self.groqai = Groq()

    def send_message(
        self,
        message: List[Any],
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        tools_param: List[ChatCompletionToolParam] | NotGiven = (
            [ChatCompletionToolParam(type="function", function=tool["function"]) for tool in tools]
            if tools
            else NOT_GIVEN
        )
        response = (
            self.groqai.chat.completions.create(
                model=self.model,
                messages=to_openai_message(message, system_message),
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools_param,
            )
            if tools
            else self.groqai.chat.completions.create(
                model=self.model,
                messages=to_openai_message(message, system_message),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.choices[0].message.content

    def stream_message(
        self,
        message: List[Any],
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[ChatCompletionChunk] | MessageStreamManager | Stream[ChatCompletionChunk]:
        pass
