from typing import Optional

from ml_boilerplate_module.llm.client_factory import get_llm_client
from ml_boilerplate_module.web.website import Website


def extract_links(url: str, provider: str = "ollama", fmt: str = "json", model: Optional[str] = None) -> str:
    website = Website(url)
    website.scrape()
    client = get_llm_client(provider, model=model) if model else get_llm_client(provider)
    return client.extract_links(website, fmt)
