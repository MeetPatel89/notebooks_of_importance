from typing import List, Optional, Tuple

from ml_boilerplate_module.llm.client_factory import get_llm_client
from ml_boilerplate_module.llm.interfaces import LLMResponse, Message
from ml_boilerplate_module.llm.vectordb import VectorDB


class Agent:
    def __init__(
        self,
        provider: str,
        model: str,
        prompt_context: Optional[str] = None,
    ):
        self._provider = provider
        self._model = model
        self._client = get_llm_client(provider=self._provider, model=self._model)
        self._vector_db: Optional[VectorDB] = None
        self._message_history: List[Message] = []
        self._prompt_context: Optional[str] = prompt_context

    def add_message(self, role: str, content: str) -> None:
        self._message_history.append(Message(role=role, content=content))

    @property
    def vector_db(self) -> Optional[VectorDB]:
        return self._vector_db

    @vector_db.setter
    def vector_db(self, vector_db: Optional[VectorDB]) -> None:
        self._vector_db = vector_db

    @property
    def message_history(self) -> List[Message]:
        return self._message_history

    @property
    def provider(self) -> str:
        return self._provider

    @provider.setter
    def provider(self, provider: str) -> None:
        self._provider = provider

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str) -> None:
        self._model = model

    def set_client(self) -> None:
        self._client = get_llm_client(provider=self._provider, model=self._model)

    def retrieve_context(self, query: str) -> List[str]:
        if self._vector_db is None:
            raise ValueError("Vector database is not set")
        results = self._vector_db.search_vectors(query)
        return [result[2] for result in results]

    def retrieve_context_with_score(self, query: str) -> List[Tuple[str, float]]:
        if self._vector_db is None:
            raise ValueError("Vector database is not set")
        results = self._vector_db.search_vectors(query)
        return [(result[2], result[1]) for result in results]

    def send_message(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        retrieve_context: bool = False,
    ) -> LLMResponse:
        prompt_context = ""
        if retrieve_context:
            context = self.retrieve_context_with_score(user_message)
            print("--------------------------------")
            print("Retreiving context...")
            for chunk, score in context:
                print("--------------------------------")
                print(f"Chunk: {chunk}")
                print(f"Score: {score}")
                print("--------------------------------")
                prompt_context += f"\n{chunk}"
            print("--------------------------------")
            print(f"Prompt context: {prompt_context}")
            print("--------------------------------")
        if user_message:
            if prompt_context:
                user_message = f"{user_message}\nHere is the retrieved context:\n{prompt_context}"
            self.add_message(role="user", content=user_message)
        print("Message history from chatbot: ")
        print(self.message_history)
        response = self._client.send_message(messages=self._message_history, system_message=system_message)
        self._message_history.append(Message(role="assistant", content=response.content))
        return response
