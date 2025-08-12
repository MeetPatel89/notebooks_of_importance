from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from .exceptions import WebsiteScrapingError


class Website:
    """Represents a website and allows extraction of title and text content."""

    def __init__(self, url: str):
        self.url = url
        self._title: Optional[str] = None
        self._text: Optional[str] = None
        self._links: List[str] = []

    @property
    def title(self) -> Optional[str]:
        return self._title

    @property
    def text(self) -> Optional[str]:
        return self._text

    @property
    def links(self) -> List[str]:
        return self._links

    def scrape(self) -> None:
        try:
            resp = requests.get(self.url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            self._title = soup.title.string if soup.title else "No title found"
            self._text = soup.body.get_text(separator="\n", strip=True) if soup.body else "No text found"
            self._links = [
                str(link["href"])
                for link in soup.find_all("a")
                if isinstance(link, Tag) and link.has_attr("href")
            ]
        except Exception as exc:
            raise WebsiteScrapingError(f"Failed to scrape {self.url}: {exc}") from exc
