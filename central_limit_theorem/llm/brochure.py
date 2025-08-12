from typing import Optional

from ml_boilerplate_module.llm.client_factory import get_llm_client
from ml_boilerplate_module.web.website import Website


def create_brochure(
    url: str,
    provider: str = "openai",
    fmt: str = "markdown",
    model: Optional[str] = None,
    company_name: Optional[str] = None,
) -> str:
    website = Website(url)
    website.scrape()
    client = get_llm_client(provider, model=model) if model else get_llm_client(provider)
    return client.create_brochure(website=website, fmt=fmt, company_name=company_name)
