import json
from typing import Optional

from ollama import ChatResponse, chat

from ml_boilerplate_module.llm.interfaces import LLMClient
from ml_boilerplate_module.llm.prompt_utils import build_user_prompt, get_system_prompt
from ml_boilerplate_module.web.website import Website


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        print(f"Using Ollama model: {self.model}")

    def summarize(self, website: Website, fmt: str = "markdown") -> str:
        messages = [
            {"role": "system", "content": get_system_prompt("web_summarizer")},
            {"role": "user", "content": build_user_prompt(website, fmt, "web_summarizer")},
        ]
        response: ChatResponse = chat(model=self.model, messages=messages)
        return str(response["message"]["content"])

    def extract_links(self, website: Website, fmt: str = "json") -> str:
        messages = [
            {"role": "system", "content": get_system_prompt("link_extractor")},
            {"role": "user", "content": build_user_prompt(website, fmt, "link_extractor")},
        ]
        response: ChatResponse = chat(model=self.model, messages=messages)
        return str(response["message"]["content"])

    def create_brochure(
        self,
        website: Website,
        fmt: str = "markdown",
        company_name: Optional[str] = None,
    ) -> str:
        print("Fetching website landing page text...")
        website_text = self.summarize(website)
        print("Fetching links...")
        links_json = self.extract_links(website)
        links_data = json.loads(links_json.split("```json")[1].split("```")[0])
        prompt_append = f"Website landing page text: {website_text}\n"
        print("Fetching link summaries...")
        for link in links_data["links"]:
            url = link["url"]
            description = link["description"]
            print(f"Fetching summary for {description}...")
            link_text = self.summarize(Website(url))
            prompt_append += f"{description}: {link_text}\n\n"
        messages = [
            {"role": "system", "content": get_system_prompt("brochure_creator")},
            {
                "role": "user",
                "content": build_user_prompt(
                    website=website,
                    fmt=fmt,
                    type="brochure_creator",
                    company_name=company_name,
                    prompt_append=prompt_append,
                ),
            },
        ]
        response: ChatResponse = chat(model=self.model, messages=messages)
        return str(response["message"]["content"])
