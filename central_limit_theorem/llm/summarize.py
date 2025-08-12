from typing import Optional

from ml_boilerplate_module.llm.client_factory import get_llm_client
from ml_boilerplate_module.web.website import Website


def summarize_website(
    url: str, provider: str = "openai", fmt: str = "json", model: Optional[str] = None
) -> str:
    website = Website(url)
    try:
        website.scrape()
        client = get_llm_client(provider, model=model) if model else get_llm_client(provider)
        return client.summarize(website, fmt)
    except Exception as e:
        print(f"Error summarizing website: {e}")
        return f"Error summarizing website: {e}"
