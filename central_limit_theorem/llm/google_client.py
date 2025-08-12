from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse

from ml_boilerplate_module.llm.interfaces import LLMClient
from ml_boilerplate_module.llm.message import Message, to_google_message


class GoogleAIClient(LLMClient):
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        print(f"Using Google model: {self.model}")
        self.googleai = genai.Client()

    def send_message(
        self,
        message: List[Any],
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        config = GenerateContentConfig(
            system_instruction=system_message or "You are a helpful assistant.",
            temperature=temperature or 0.7,
        )
        response = self.googleai.models.generate_content(
            model=self.model,
            config=config,
            contents=to_google_message(message),
        )
        return str(response.text)

    def stream_message(
        self,
        message: List[Message],
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[GenerateContentResponse]:
        config = GenerateContentConfig(
            system_instruction=system_message or "You are a helpful assistant.",
            temperature=temperature or 0.7,
        )
        response = self.googleai.models.generate_content_stream(
            model=self.model,
            config=config,
            contents=to_google_message(message),
        )
        return response

    # def summarize(self, website: Website, fmt: str = "markdown") -> str:
    #     return self.send_message(
    #         system_message=get_system_prompt("web_summarizer"),
    #         user_message=build_user_prompt(website, fmt, "web_summarizer"),
    #     )

    # def extract_links(self, website: Website, fmt: str = "json") -> str:
    #     return self.send_message(
    #         system_message=get_system_prompt("link_extractor"),
    #         user_message=build_user_prompt(website, fmt, "link_extractor"),
    #     )

    # def create_brochure(
    #     self,
    #     website: Website,
    #     fmt: str = "markdown",
    #     company_name: Optional[str] = None,
    # ) -> str:
    #     print("Fetching website landing page text...")
    #     website_text = self.summarize(website)
    #     print("Fetching links...")
    #     links_json = self.extract_links(website)
    #     links_data = json.loads(links_json.split("```json")[1].split("```")[0])
    #     prompt_append = f"Website landing page text: {website_text}\n"
    #     print("Fetching link summaries...")
    #     for link in links_data["links"]:
    #         url = link["url"]
    #         description = link["description"]
    #         print(f"Fetching summary for {description}...")
    #         link_text = self.summarize(Website(url))
    #         prompt_append += f"{description}: {link_text}\n\n"

    #     return self.send_message(
    #         system_message=get_system_prompt("brochure_creator"),
    #         user_message=build_user_prompt(
    #             website=website,
    #             fmt=fmt,
    #             type="brochure_creator",
    #             company_name=company_name,
    #             prompt_append=prompt_append,
    #         ),
    #     )
