from typing import Any, List

import numpy as np
import numpy.typing as npt
import tiktoken
from nltk.tokenize import sent_tokenize, word_tokenize  # type: ignore
from openai import embeddings

from ml_boilerplate_module.config import load_config

# nltk.download("punkt_tab")
# nltk.download("punkt")


def split_sentences(text: str) -> Any:
    return sent_tokenize(text)


def split_words(text: str) -> Any:
    return word_tokenize(text)


encoding = tiktoken.get_encoding("cl100k_base")  # For GPT-4o, GPT-4, GPT-3.5-turbo


def num_tokens(text: str) -> int:
    return len(encoding.encode(text))


def chunk_text_by_tokens(text: str, max_tokens: int = 256) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    max_token_sentence = 0
    for sentence in sentences:
        if num_tokens(sentence) > max_token_sentence:
            max_token_sentence = num_tokens(sentence)
        if num_tokens(current_chunk + sentence) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def embed_text(text: str) -> npt.NDArray[np.float64] | None:
    response = embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding) if response.data else None


def cosine_similarity(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> Any:
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))


def euclidean_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> Any:
    print("Norm of a: ", np.linalg.norm(a, axis=1))
    print("Norm of b: ", np.linalg.norm(b))
    return np.linalg.norm(a - b, axis=1)


def inner_product(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> Any:
    return a @ b


if __name__ == "__main__":
    load_config()
    given_text = "Hello, how are you?"
    print(embed_text(given_text))
    print("------------------------------------")
    print("Embedding again")
    print(embed_text(given_text))
