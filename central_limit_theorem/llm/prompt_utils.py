from typing import Dict, Optional

from ml_boilerplate_module.web.website import Website


def get_system_prompt(type: str) -> str:
    if type == "web_summarizer":
        return (
            "You are an assistant that analyzes the contents of a website and provides a short summary, "
            "ignoring text that might be navigation related. The websites I am analyzing is about AI "
            "companies and AI products."
            "The output should be concise and to the point. "
            "The output should be in the format specified in the user prompt."
        )
    elif type == "link_extractor":
        return (
            "You are in charge of creating a marketing brochure for a company "
            "You are provided with a list of links found on a website. "
            "You are able to decide which of the links would be most "
            "relevant to include in a brochure about the company, "
            "such as links to an About page, a Contact page, a Blog, "
            "a Products page, Company Page, Careers/Jobs pages, etc."
        )
    elif type == "brochure_creator":
        return (
            "You are an assistant that analyzes the contents of serveral relevant pages "
            "from a company's website and creates a detailed marketing brochure about the company "
            "for prospective clients, investors and recruits. Include details about company's "
            "culture, customers, careers/jobs, products, services, etc. or any relevant information "
            "you can find in the content of the pages."
        )
    else:
        raise ValueError(f"Invalid type: {type}")


def get_response_format_prompt(fmt: str) -> str:
    fmt = fmt.lower()
    mapping = {
        "markdown": "Please provide the response in markdown format like so: ```markdown\n...\n```",
        "html": "Please provide the response in HTML format like so: ```html\n...\n```",
        "json": """
        Please provide the response in JSON format ensuring your response is concise, accurate, 
        and adheres strictly to the below structure.""",
        "yaml": "Please provide the response in YAML format like so: ```yaml\n...\n```",
    }
    return mapping.get(fmt, "Please provide the response in plain text.")


def set_response_format_examples(type: str) -> Dict[str, str]:
    examples = {}
    if type == "web_summarizer":
        examples = {
            "markdown": """
Example:
```markdown
# Website Title
Brief description of the website's main purpose.

## Key Features
- Feature 1: Description
- Feature 2: Description

## Main Content
Summary of the primary content and value proposition.
```""",
            "html": """
Example:
```html
<h1>Website Title</h1>
<p>Brief description of the website's main purpose.</p>
<h2>Key Features</h2>
<ul>
    <li><strong>Feature 1:</strong> Description</li>
    <li><strong>Feature 2:</strong> Description</li>
</ul>
<h2>Main Content</h2>
<p>Summary of the primary content and value proposition.</p>
```""",
            "json": """
{
    "title": "Website Title",
    "description": "Brief description of the website's main purpose",
    "features": [
        {"name": "Feature 1", "description": "Description"},
        {"name": "Feature 2", "description": "Description"},
        ...
    ],
    "summary": "Summary of the primary content and value proposition"
}
""",
            "yaml": """
Example:
```yaml
title: Website Title
description: Brief description of the website's main purpose
features:
  - name: Feature 1
    description: Description
  - name: Feature 2
    description: Description
summary: Summary of the primary content and value proposition
```""",
        }
    elif type == "link_extractor":
        examples = {
            "json": """
Example:
```json
{
    "links": [
        {
            "url": "https://www.google.com",
            "description": "Google's main website"
        },
        {
            "url": "https://www.google.com/about",
            "description": "Google's About page"
        }
    ]
}
```""",
        }
    else:
        raise ValueError(f"Invalid type: {type}")

    return examples


def get_response_format_examples(fmt: str, type: str) -> str:
    """Provides diverse examples for each response format to guide the LLM."""
    fmt = fmt.lower()

    examples = set_response_format_examples(type)

    return examples.get(
        fmt, "Provide a clear, concise text summary of the website's main content and purpose."
    )


def build_user_prompt(
    website: Website,
    fmt: str,
    type: str,
    company_name: Optional[str] = None,
    prompt_append: Optional[str] = None,
) -> str:
    if type == "web_summarizer":
        return (
            f"You are looking at a website titled {website.title}\n"
            "The contents of this website are as follows:\n"
            "Please provide a short summary of the contents of this website, "
            "ignoring text that might be navigation related and focusing on the main content. "
            f"{get_response_format_examples(fmt, type)}\n"
            f"{website.text if website.text else 'No text found'}"
        )
    elif type == "link_extractor":
        newline = "\n"
        links_text = newline.join(website.links)
        return (
            f"Here is the list of links on the website of {website.url} - please decide which of the links "
            f"would be most relevant to include in a brochure about the company, "
            f"respond with full https URL. Do not include Terms of Service, Privacy, email links.\n"
            f"Links (some might be relative links):\n{links_text}\n"
            f"{get_response_format_prompt(fmt)}\n"
            f"{get_response_format_examples(fmt, type)}"
        )
    elif type == "brochure_creator":
        return (
            f"You are looking at a company called {company_name}\n"
            "Here are the contents of the company's landing page as well as "
            "other rlevant pages. Create a detailed marketing brochure about the company.\n"
            "Include hyperlinks to the relevant pages in the brochure."
            "These hyperlinks should be directed to the relevant pages on the company's website "
            "and not to home page.\n"
            f"{prompt_append}\n{get_response_format_prompt(fmt)}\n"
            if prompt_append
            else f"{get_response_format_prompt(fmt)}\n"
        )
    else:
        raise ValueError(f"Invalid type: {type}")
