from typing import Any

from ml_boilerplate_module.llm.interfaces import LLMClient
from ml_boilerplate_module.llm.message import Message
from ml_boilerplate_module.llm.prompt_utils import build_user_prompt, get_system_prompt
from ml_boilerplate_module.web.website import Website

summarize_website_function = {
    "name": "summarize_website",
    "description": (
        "Summarize a website. Call this function whenever you want to scrape a website or a url link"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL of the website to summarize"},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}


def summarize(client: LLMClient, website: Website, fmt: str = "markdown") -> Any:
    return client.send_message(
        system_message=get_system_prompt("web_summarizer"),
        message=[
            Message(role="user", content=build_user_prompt(website, fmt, "web_summarizer")),
        ],
    )


def summarize_website(url: str, client: LLMClient, fmt: str = "json") -> Any:
    website = Website(url)
    try:
        website.scrape()
        return summarize(client, website, fmt)
    except Exception as e:
        return f"Error summarizing website: {e}"
