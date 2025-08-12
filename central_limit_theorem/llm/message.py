from typing import Any, List, Literal, Optional, cast

from anthropic.types.message_param import MessageParam
from google.genai.types import Content, Part
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from ml_boilerplate_module.llm.interfaces import Message


def to_openai_message(
    messages: List[Message], system_message: Optional[str] = None
) -> List[ChatCompletionMessageParam]:
    client_message: List[ChatCompletionMessageParam] = [
        (
            ChatCompletionSystemMessageParam(role="system", content=system_message)
            if system_message
            else ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant.")
        )
    ]

    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, str):
            client_message.append(ChatCompletionUserMessageParam(role="user", content=msg.content))
        elif msg.role == "assistant" and isinstance(msg.content, str):
            client_message.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg.content))
    return client_message


def to_google_message(messages: List[Message]) -> List[Content | Any]:
    return [Content(role=msg["role"], parts=[Part(text=msg["content"])]) for msg in messages]


def to_anthropic_message(messages: List[Message]) -> List[MessageParam]:
    client_message: List[MessageParam] = []
    for msg in messages:
        if msg.role not in ["user", "assistant"]:
            raise ValueError(f"Only user and assistant messages are supported for Anthropic, got {msg.role}")
        else:
            client_message.append(
                MessageParam(
                    role=cast(Literal["user", "assistant"], msg.role),
                    content=msg.content,
                )
            )
    return client_message
