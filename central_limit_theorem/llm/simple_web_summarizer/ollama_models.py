from typing import Dict, List

from ollama import ChatResponse, chat

from ml_boilerplate_module.llm.simple_web_summarizer.website import Website

system_prompt = (
    "You are an assistant that analyzes the contents of a website and provides a short summary, "
    "ignoring text that might be navigation related. The website I am analyzing is about large "
    "language models. Respond in markdown format."
)


def response_format(format: str) -> str:
    if format == "markdown":
        return "Please provide the response in markdown format."
    elif format == "html":
        return "Please provide the response in HTML format."
    elif format == "json":
        return "Please provide the response in JSON format."
    elif format == "yaml":
        return "Please provide the response in YAML format."
    else:
        return "Please provide the response in plain text."


def user_prompt(website: Website, format: str) -> str:
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += (
        "\nThe contents of this website are as follows:"
        "\nPlease provide a short summary of the contents of this website, "
        "ignoring text that might be navigation related and "
        f"focusing on the main content. {response_format(format)}\n"
    )
    user_prompt += website.text if website.text else "No text found"
    return user_prompt


def get_msg(url: str, format: str = "markdown") -> List[Dict[str, str]]:
    website = Website(url)
    website.scrape()
    msg = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt(website, format),
        },
    ]
    return msg


def summarize(url: str, format: str = "markdown", model: str = "llama3.2") -> str:
    messages = get_msg(url, format)
    response: ChatResponse = chat(
        model=model,
        messages=messages,
    )
    return str(response["message"]["content"])
