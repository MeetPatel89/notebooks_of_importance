import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import numpy.typing as npt
from chromadb.api.types import QueryResult
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from ml_boilerplate_module.config import load_config
from ml_boilerplate_module.llm.doc_preprocessor import extract_and_chunk_mds, extract_and_chunk_pdfs
from ml_boilerplate_module.llm.nlp_utils import embed_text, inner_product


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]


class VectorDB(ABC):
    @abstractmethod
    def add_vector(
        self, embedding: npt.NDArray[np.float64] | None = None, metadata: Optional[str] = None
    ) -> None: ...

    @abstractmethod
    def search_vectors(self, user_query: str, k: int = 5) -> List[Tuple[int, float, str]] | QueryResult: ...

    @abstractmethod
    def load_documents(self, repo_path: str) -> None: ...


class ChromaVectorDB(VectorDB):
    def __init__(self, db_path: str):
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name="chroma_collection",
            embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),  # type: ignore
            configuration={"hnsw": {"space": "l2"}},
        )

    def add_vector(
        self, embedding: npt.NDArray[np.float64] | None = None, metadata: Optional[str] = None
    ) -> None:
        if metadata is None:
            raise ValueError("Metadata is required")
        parsed_metadata = json.loads(metadata)
        metadata_dict = {}
        for key in parsed_metadata:
            if isinstance(parsed_metadata[key], list):
                continue
            metadata_dict[key] = parsed_metadata[key]
        self._collection.add(
            ids=[parsed_metadata["chunk_id"]],
            documents=[parsed_metadata["text"]],
            metadatas=[metadata_dict],
        )

    def chunk_documents(self, folder_path: str, output_dir: str, file_type: Optional[str] = None) -> None:
        if file_type == "pdf":
            extract_and_chunk_pdfs(folder_path, output_dir)
        elif file_type == "md":
            extract_and_chunk_mds(folder_path, output_dir)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def load_documents(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                # Metadata can be the entire chunk as a JSON string (for context retrieval)
                metadata_str = json.dumps(chunk, ensure_ascii=False)
                self.add_vector(metadata=metadata_str)

    def search_vectors(self, user_query: str, k: int = 5) -> List[Tuple[int, float, str]] | QueryResult:
        results = self._collection.query(
            query_texts=[user_query],
            n_results=k,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        return results

    def close(self) -> None:
        pass


# Implement VectorDB using SQLITE
class SqliteVectorDB(VectorDB):
    def __init__(self, db_path: str, embed_fn: Callable[[str], npt.NDArray[np.float64] | None]):
        self.conn = sqlite3.connect(db_path)
        self._cursor = self.conn.cursor()
        self.embed_fn = embed_fn
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.conn.commit()

    def add_vector(
        self, embedding: npt.NDArray[np.float64] | None = None, metadata: Optional[str] = None
    ) -> None:
        if embedding is None:
            if metadata is None:
                raise ValueError("Metadata is required")
            embedding = self.embed_fn(metadata)
        self._cursor.execute(
            "INSERT INTO vectors (embedding, metadata) VALUES (?, ?)",
            (embedding.tobytes() if embedding is not None else None, metadata),
        )
        self.conn.commit()

    def search_vectors(self, user_query: str, k: int = 5) -> List[Tuple[int, float, str]] | QueryResult:
        """Search for k most similar vectors using cosine similarity.
        Returns list of tuples containing (id, similarity_score, metadata)
        """
        query_embedding = self.embed_fn(user_query)
        if query_embedding is None:
            raise ValueError("Query embedding is None")
        # Get all vectors from db
        self._cursor.execute("SELECT id, embedding, metadata FROM vectors")
        rows = self._cursor.fetchall()

        if not rows:
            return []

        # Convert BLOB embeddings back to numpy arrays
        ids = []
        embeddings = []
        metadata = []
        for row in rows:
            ids.append(row[0])
            embeddings.append(np.frombuffer(row[1], dtype=np.float64))
            metadata.append(row[2])

        # Stack embeddings into a matrix
        embedding_matrix = np.vstack(embeddings)

        # Calculate similarities
        # similarities = cosine_similarity(embedding_matrix, query_embedding)
        # similarities = euclidean_distance(embedding_matrix, query_embedding)
        similarities = inner_product(embedding_matrix, query_embedding)

        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        # top_k_indices = np.argsort(similarities)[:k]

        # Return results
        results = [(ids[i], embeddings[i], float(similarities[i]), metadata[i]) for i in top_k_indices]

        return results

    def chunk_documents(self, folder_path: str, output_dir: str, file_type: Optional[str] = None) -> None:
        if file_type == "pdf":
            extract_and_chunk_pdfs(folder_path, output_dir)
        elif file_type == "md":
            extract_and_chunk_mds(folder_path, output_dir)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def load_documents(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunk_text = chunk["text"]
                # Metadata can be the entire chunk as a JSON string (for context retrieval)
                metadata_str = json.dumps(chunk, ensure_ascii=False)
                self.add_vector(self.embed_fn(chunk_text), metadata_str)

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    load_config()
    vector_db_sqlite = SqliteVectorDB(
        db_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset/bio_vector_db.db",
        embed_fn=embed_text,
    )
    print("SQLite Vector DB initialized...")
    # chunk markdown files
    # vector_db_sqlite.chunk_documents(
    #     folder_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo\markdown",
    #     output_dir=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo/chunks",
    #     file_type="md",
    # )
    # # chunk pdf files
    # vector_db.chunk_documents(
    #     folder_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo\pdf",
    #     output_dir=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo/chunks",
    #     file_type="pdf",
    # )
    # load documents
    # vector_db_sqlite.load_documents(
    #     json_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo/chunks/all_chunks.jsonl"
    # )

    results_sqlite = vector_db_sqlite.search_vectors(
        user_query="Please summarize what HealthGenLLM does?",
        k=3,
    )
    print("--------------------------------")
    print("Results from SQLite Vector DB:")
    print("--------------------------------")
    print(results_sqlite)
    print("--------------------------------")
    vector_db_sqlite.close()
    print("SQLite Vector DB closed...")
    vector_db_chroma = ChromaVectorDB(
        db_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset/bio_vector_db.chroma"
    )
    print("Chroma Vector DB initialized...")
    # vector_db_chroma.load_documents(
    #     json_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\private_repo/chunks/all_chunks.jsonl"
    # )
    results_chroma = vector_db_chroma.search_vectors(
        user_query="Please summarize what HealthGenLLM does?",
        k=3,
    )
    print("--------------------------------")
    print("Results from Chroma Vector DB:")
    print("--------------------------------")
    print(results_chroma)
    print("--------------------------------")
    # for result in results["documents"][0]:
    #     print("--------------------------------")
    #     print(result)
    #     print("--------------------------------")
    vector_db_chroma.close()
    print("Chroma Vector DB closed...")
