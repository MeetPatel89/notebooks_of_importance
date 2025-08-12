website_json_format = """
**Example Output Format:**:
json should have the following keys:
- title (string)
- description (string)
- features (array of objects) with the following keys:
  - name (string)
  - description (string)
- summary (string) should be summary of the website

**Example Output:**:
{
    "title": "Website Title",
    "description": "Brief description of the website's main purpose",
    "features": [
        {"name": "Feature 1", "description": "Description"},
        {"name": "Feature 2", "description": "Description"},
        ...
    ],
    "detailed_summary": "Detailed summary of the primary content and value proposition"
}
"""

research_assistant_2_json_format = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the report."},
        "introduction": {"type": "string", "description": "The introduction section of the report."},
        "methodology": {
            "type": "string",
            "description": "The methodology section describing how the information was gathered or analyzed.",
        },
        "results": {
            "type": "array",
            "description": "A list of company reports.",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Name of the company."},
                    "description": {"type": "string", "description": "Short description of the company."},
                    "company_website": {
                        "type": "string",
                        "format": "uri",
                        "description": "Website URL of the company.",
                    },
                    "company_summary": {
                        "type": "string",
                        "description": "A summary of the company's role or focus in AI.",
                    },
                    "products": {
                        "type": "array",
                        "description": "A list of products offered by the company.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Name of the product."},
                                "description": {
                                    "type": "string",
                                    "description": "Short description of the product.",
                                },
                                "product_website": {
                                    "type": "string",
                                    "format": "uri",
                                    "description": "Website URL of the product.",
                                },
                            },
                            "required": ["name", "description", "product_website"],
                        },
                    },
                },
                "required": ["company", "description", "company_website", "company_summary", "products"],
            },
        },
        "conclusion": {"type": "string", "description": "The conclusion section of the report."},
        "references": {
            "type": "array",
            "description": "List of references cited in the report.",
            "items": {"type": "string"},
        },
        "appendices": {
            "type": "array",
            "description": "List of appendices for additional information.",
            "items": {"type": "string"},
        },
    },
    "required": ["title", "introduction", "methodology", "results", "conclusion", "references", "appendices"],
}
