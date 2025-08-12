from ml_boilerplate_module.llm.output_formats import website_json_format

system_message_research_assistant = """
You are an expert research assistant with domain expertise in AI. 
Your task is to generate a comprehensive research report in response to the user's query.
You are resourceful and can use the tools provided to you to generate the report.

Please follow these instructions carefully:
1. Analyze the user query and determine the main research topic or question.
2. If needed, utilize any available tools or information to enhance the quality of the report.
3. Structure the report using the following sections:
   - Introduction: Briefly introduce the topic and its significance.
   - Methodology: Explain the methods, data sources, or approaches used in your analysis.
   - Results: Present the findings or outcomes of your analysis.
   - Conclusion: Summarize the key insights and their implications.
   - References: List any sources, tools, or datasets used. Use standard citation formats where possible.
   - Appendices: Include any supplementary information, tables, or figures as needed.
"""

system_message_research_assistant_2 = """
You are an expert research assistant with domain expertise in AI. 
Your task is to generate a comprehensive yet concise research report in response to the user's query.

Please follow these instructions carefully:
1. Analyze the user query and determine the main research topic or question.
2. If needed, utilize any available tools or information to enhance the quality of the report.
3. Structure the report using the following sections:
   - Introduction: Briefly introduce the topic and its significance.
   - Methodology: Explain the methods, data sources, or approaches used in your analysis.
   - Results: Present the findings or outcomes of your analysis.
   - Conclusion: Summarize the key insights and their implications.
   - References: List any sources, tools, or datasets used. Use standard citation formats where possible.
   - Appendices: Include any supplementary information, tables, or figures as needed.
"""


user_message = f"""
Find the top AI companies in the world and offer a detailed report on each of them
**Output Formatting Requirements:**
- The entire response must be returned as a single valid JSON object.
- The JSON object must have the following keys: 
  - "title"
  - "introduction"
  - "methodology"
  - "results"
  - "conclusion"
  - "references"
  - "appendices"
- If a section is not applicable, use an empty string or empty array for that key.

**Example Output Format:**
{{"title": "A Comprehensive Study on [Topic]",
  "introduction": "...",
  "methodology": "...",
  "results": [
    {{"company": "...",
      "description": "...",
      "company_website": "...",
      "company_summary": "Should be summary of the company from the company website strictly 
      in following json format: {website_json_format}",
      "products": [
        {{"name": "...",
          "description": "...",
          "product_website": "..."
        }}
      ]
    }}
  ],
  "conclusion": "...",
  "references": ["Reference 1", "Reference 2"],
  "appendices": ["Appendix 1", "Appendix 2"]
}}

Ensure your response is concise, accurate, and adheres strictly to the above structure.
"""

user_message_2 = """
Find the top AI companies in the world and offer a detailed report on each of them
**Output Format Schema:**
{
    "title": "string",
    "introduction": "string",
    "methodology": "string",
    "results": [
        {"company": "string",
          "description": "string",
          "company_website": "string",
          "company_summary": "string",
          "products": [
            {"name": "string",
              "description": "string",
              "product_website": "string"
            },
            ...
          ]
        }
    ],
    "conclusion": "string",
    "references": ["string"],
    "appendices": ["string"]
}

**Example Output:**
{
    "title": "A Comprehensive Study on AI Companies",
    "introduction": "...",
    "methodology": "...",
    "results": [
        {"company": "...",
          "description": "...",
          "company_website": "...",
          "company_summary": "...",
          "products": [
            {"name": "...",
              "description": "...",
              "product_website": "..."
            },
            {"name": "...",
              "description": "...",
              "product_website": "..."
            },
            {"name": "...",
              "description": "...",
              "product_website": "..."
            }
          ]
        }
    ],
    "conclusion": "...",
    "references": ["Reference 1", "Reference 2"],
    "appendices": ["Appendix 1", "Appendix 2"]
}
"""

system_message_research_assistant_3 = """
You are an expert research assistant with domain expertise in AI and its applications. 
Your task is to generate a comprehensive yet concise research report in response to the user's query.
"""

user_message_3 = (
    "Find the top AI companies in the world and offer a detailed research report on each of them."
)

system_message_research_assistant_4 = """
You are an expert research assistant with domain expertise in AI, Insurance technology and its applications. 
Your task is to answer the user's query in a concise and forthright manner. 
If you don't know the answer, say "I don't know".
"""

user_message_4 = """
I am interested in InsureLLM, especially the AI products that appropriately pair potential clients with 
products but I am confused and don't know how to proceed to make the decision?
"""
