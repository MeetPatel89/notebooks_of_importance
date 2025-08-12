from typing import Union

import requests
from bs4 import BeautifulSoup


class Website:
    def __init__(
        self,
        url: str,
        title: Union[str, None] = None,
        text: Union[str, None] = None,
    ):
        self.url = url
        self._title = title
        self._text = text

    @property
    def title(self) -> Union[str, None]:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value

    @property
    def text(self) -> Union[str, None]:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value

    def scrape(self) -> None:
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        self.text = soup.body.get_text(separator="\n", strip=True) if soup.body else "No text found"
