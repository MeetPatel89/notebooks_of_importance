from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    raw: Any = None


class LLMClient(ABC):
    @abstractmethod
    def send_message(
        self,
        messages: List[Message],
        system_message: Optional[str] = None,
    ) -> LLMResponse: ...
