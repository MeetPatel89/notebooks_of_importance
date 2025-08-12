import json

import gradio as gr
import numpy as np
from openai.types.chat import ChatCompletion

from ml_boilerplate_module import load_config
from ml_boilerplate_module.file_utils import read_all_text_files_recursive
from ml_boilerplate_module.llm.client_factory import get_llm_client
from ml_boilerplate_module.llm.groq_client import GroqAIClient
from ml_boilerplate_module.llm.interfaces import Message
from ml_boilerplate_module.llm.nlp_utils import chunk_text_by_tokens, cosine_similarity, embed_text
from ml_boilerplate_module.llm.openai_client import OpenAIClient
from ml_boilerplate_module.llm.output_formats import research_assistant_2_json_format
from ml_boilerplate_module.llm.prompts import (
    system_message_research_assistant,
    system_message_research_assistant_3,
    system_message_research_assistant_4,
    user_message,
    user_message_3,
    user_message_4,
)
from ml_boilerplate_module.llm.tools import summarize_website, summarize_website_function
from ml_boilerplate_module.llm.vectordb import SqliteVectorDB, VectorDB

load_config()


tools = [{"type": "function", "function": summarize_website_function}]


# tools
def main() -> None:
    print("Starting the AI application...")
    message_history = [{"role": "user", "content": user_message}]
    client = OpenAIClient(model="gpt-4.1")
    response = client.send_message(
        message=message_history, system_message=system_message_research_assistant, tools=tools
    )
    print("First response from chatbot: ")
    print("--------------------------------")
    print(response)
    print("--------------------------------")
    if isinstance(response, ChatCompletion):
        if response.choices[0].finish_reason == "tool_calls":
            print("AI model called tools...")
            print("--------------------------------")
            print("Message for tool calls: ")
            print(response.choices[0].message)
            message_history.append(response.choices[0].message)  # type: ignore
            print("--------------------------------")
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    print("--------------------------------")
                    print(f"Tool call: {tool_call}")
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    tool_id = tool_call.id
                    print(f"Tool id: {tool_id}")
                    print(f"Tool name: {tool_name}")
                    print(f"Tool arguments: {tool_args}")
                    if tool_name == "summarize_website":
                        website_url = json.loads(tool_args).get("url")
                        print(f"Website url: {website_url}")
                        summary = summarize_website(website_url, client)
                        print(f"Summary: {summary}")
                        summary_extract = None
                        if isinstance(summary, ChatCompletion):
                            summary_extract = summary.choices[0].message.content
                        else:
                            summary_extract = summary
                        message_history.append(
                            {"role": "tool", "content": summary_extract, "tool_call_id": tool_id}  # type: ignore
                        )
            print("--------------------------------")
            print("Message history: ")
            print(message_history)
            print("--------------------------------")
            print("Sending message to chatbot along with the tool call response...")
            response = client.send_message(
                message=message_history, system_message=system_message_research_assistant
            )
            print("Response from chatbot: ")
            print("--------------------------------")
            print(response.choices[0].message.content)
            print("--------------------------------")


# structured output
def main2() -> None:
    print("Starting the AI application from main2...")
    message_history = [{"role": "user", "content": user_message_3}]
    client = OpenAIClient(model="gpt-4.1")
    response = client.send_message(
        message=message_history,
        system_message=system_message_research_assistant_3,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "research_report", "schema": research_assistant_2_json_format},
        },
    )
    print("Response from chatbot: ")
    print("--------------------------------")
    print(response.choices[0].message.content)
    print("--------------------------------")


# numpy vector search
def main3() -> None:
    print("Starting the AI application from main3...")
    repo_context = read_all_text_files_recursive("./dataset/repository/")
    chunks = []
    for file_content in repo_context.values():
        chunks.extend(chunk_text_by_tokens(file_content))
    embeddings = [
        {
            "text": chunk,
            "embedding": embed_text(chunk),
        }
        for chunk in chunks
    ]
    embedding_matrix = np.array([embedding["embedding"] for embedding in embeddings])
    user_message_4_embedding = embed_text(user_message_4)
    similarity_scores = cosine_similarity(embedding_matrix, user_message_4_embedding)  # type: ignore
    print(np.argsort(similarity_scores)[-10:][::-1])
    print(similarity_scores)
    top_indices = np.argsort(similarity_scores)[-10:][::-1]
    print([embeddings[i] for i in top_indices])
    print([embeddings[i]["text"] for i in top_indices])

    retrieved_context = "\n".join([embeddings[i]["text"] for i in top_indices])

    user_message = (
        f"{user_message_4}\nHere is the retrieved information that might help you: {retrieved_context}"
    )
    message_history = [{"role": "user", "content": user_message}]
    client = OpenAIClient(model="gpt-4.1")
    response = client.send_message(
        message=message_history,
        system_message=system_message_research_assistant_4,
    )
    print("Response from chatbot: ")
    print("--------------------------------")
    # print(response.choices[0].message.content)
    print(response)
    print("--------------------------------")


# sqlite vector search
def main4() -> None:
    print("Starting the AI application from main4...")
    client = OpenAIClient(model="gpt-4.1")
    db = VectorDB(db_path="vectors.db")
    db.load_documents_from_folder(r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository")
    user_message_4_embedding = embed_text(user_message_4)
    if user_message_4_embedding is None:
        raise ValueError("User message 4 embedding is None")
    results = db.search_vectors(user_message_4_embedding, k=10)
    print(results)
    retrieved_context = "\n".join([result[2] for result in results])
    if retrieved_context is None:
        raise ValueError("Retrieved context is None")
    user_message = (
        f"{user_message_4}\nHere is the retrieved information that might help you: {retrieved_context}"
    )
    response = client.send_message(
        message=[{"role": "user", "content": user_message}],
        system_message=system_message_research_assistant_4,
    )
    print("Response from chatbot: ")
    print("--------------------------------")
    print(response)
    print("--------------------------------")


# groq client
def main5() -> None:
    print("Starting the AI application from main5...")

    client = GroqAIClient(model="deepseek-r1-distill-llama-70b")
    system_message = """You are a biotech professor teaching a graduate class on machine learning 
    and biology underlying it.
You always delve into details both of concepts and mathematical formulations/notations.
Respond in markdown format. For mathematical notations:
- Use $$....$$ for display equations (not \\[...\\])
- Use $....$ for inline equations"""
    # user_message = """
    # Explain coordinate vectors in linear algebra.
    # """
    # response = client.send_message(
    #     message=[{"role": "user", "content": user_message}],
    #     system_message=system_message,
    # )

    def convert_latex_delimiters(text: str) -> str:
        """Convert LaTeX delimiters to ones that work with Gradio markdown."""
        # Replace display math delimiters
        text = text.replace("\\[", "$$")
        text = text.replace("\\]", "$$")
        # Replace inline math delimiters if needed
        text = text.replace("\\(", "$")
        text = text.replace("\\)", "$")
        # Clean up any think tags that might be in the response
        if "<think>" in text:
            text = text.split("</think>")[1] if "</think>" in text else text
        return text

    def send_message_to_ai(message: str) -> str:
        vdb = VectorDB("my_vectordb.sqlite3")
        # 2. Path to your chunks
        jsonl_path = (
            "D:/projects/machine_learning_workspace/ml-boilerplate/dataset/repository/"
            "PDFs/explore_pdfs/pdf_extracted_images/all_chunks.jsonl"
        )

        vdb.load_documents_from_jsonl(jsonl_path)
        query_embedding = embed_text(message)
        if query_embedding is None:
            raise ValueError("Query embedding is None")
        message = f"{message}\nHere is the retrieved information that might help you: "
        top_chunks = vdb.search_vectors(query_embedding, k=5)
        for id, score, metadata_json in top_chunks:
            metadata = json.loads(metadata_json)
            print("--------------------------------")
            print(f"Doc: {metadata['doc_id']}, Page: {metadata['page_number']}, Score: {score:.3f}")
            print("Text snippet:", metadata["text"])  # Print start of text
            message += f"\n{metadata['text']}"
            print("--------------------------------")
        response = client.send_message(
            message=[{"role": "user", "content": message}],
            system_message=system_message,
        )
        print("--------------------------------")
        print(response)
        print("--------------------------------")
        vdb.close()
        return convert_latex_delimiters(str(response))

    gr.Interface(
        fn=send_message_to_ai,
        inputs=["text"],
        outputs=[
            gr.Markdown(
                value="Ask a question about machine learning and mathematics...",
                sanitize_html=False,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
        ],
        flagging_mode="never",
        title="Math Professor",
        description="Ask me anything about machine learning and mathematics underlying it.",
        css="""
        .gradio-container {font-size: 16px} 
        .markdown-style img {max-width: 100%} 
        .markdown-style { line-height: 1.5; padding: 15px; }
        """,
    ).launch()


def main6() -> None:
    print("Starting the AI application from main6...")
    client = get_llm_client(provider="grok", model="grok-4")
    message = [Message(role="user", content="What is an invertible matrix?")]
    response = client.send_message(
        messages=message,
        system_message=(
            "You are a helpful assistant that can answer questions about the machine learning"
            " and math in detail."
        ),
    )
    print(response)


def main7() -> None:
    vector_db = SqliteVectorDB(
        db_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset/vector_db.db",  # TODO: change to relative path
        embed_fn=embed_text,
    )
    # vector_db.load_documents(
    #     json_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset/repository/company/md_extractions/all_chunks.jsonl"
    # )
    rows = vector_db._cursor.execute("SELECT * FROM vectors")
    print(rows.fetchall())
    vector_db.close()


if __name__ == "__main__":
    main7()
