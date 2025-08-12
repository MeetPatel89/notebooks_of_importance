import re
from typing import Any, Dict, List

import gradio as gr

from ml_boilerplate_module import load_config
from ml_boilerplate_module.llm.chat import Agent
from ml_boilerplate_module.llm.interfaces import LLMResponse, Message
from ml_boilerplate_module.llm.nlp_utils import embed_text
from ml_boilerplate_module.llm.vectordb import SqliteVectorDB

# Alternative: Custom theme approach
# You can also create a custom theme for more consistent styling
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

load_config()

print("Instantiating vector db...")
chatbot = Agent(provider="openai", model="gpt-4o")

CLIENT_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-4.5-preview", "o3-mini"],
    "Anthropic": [
        "claude-opus-4-0",
        "claude-sonnet-4-0",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "Grok": ["grok-4", "grok-3", "grok-3-mini", "grok-3-fast"],
}


def update_models(client: str) -> Dict[str, Any]:
    print("Client update triggered...")
    print(f"Client updated to: {client}")
    print(f"Client models available: {CLIENT_MODELS[client]}")
    return gr.update(choices=CLIENT_MODELS[client], value=CLIENT_MODELS[client][0])


def select_model(client: str, model: str) -> Dict[str, Any]:
    print("Model update triggered...")
    print(f"Model updated to: {model}")
    return gr.update(choices=CLIENT_MODELS[client], value=model)


def select_retrieval(retrieval: str) -> Dict[str, Any]:
    print("Retrieval update triggered...")
    print(f"Retrieval updated to: {retrieval}")
    return gr.update(choices=["Off", "On"], value=retrieval)


def show_history_toggle(show_sidebar: bool) -> Dict[str, Any]:
    """Toggle sidebar visibility based on checkbox."""
    return gr.update(visible=show_sidebar)


def fix_latex_delimiters(text: str) -> str:
    """Fix LaTeX delimiters to work properly with Gradio markdown."""
    # Convert double backslashes to single backslashes for LaTeX delimiters
    text = text.replace("\\\\(", "\\(")
    text = text.replace("\\\\)", "\\)")
    text = text.replace("\\\\[", "\\[")
    text = text.replace("\\\\]", "\\]")

    # Also handle escaped newlines that might interfere with LaTeX
    text = text.replace("\\n", "\n")

    return text


def enhance_ai_response(response: str) -> str:
    """Post-process AI response to add visual enhancements while preserving markdown."""
    if not response:
        return response

    # Fix LaTeX delimiters first
    response = fix_latex_delimiters(response)

    # Keep code blocks as standard markdown - no JavaScript needed

    # Add icons to headers
    response = re.sub(r"^# (.+)$", r"# 📋 \1", response, flags=re.MULTILINE)
    response = re.sub(r"^## (.+)$", r"## 🔍 \1", response, flags=re.MULTILINE)
    response = re.sub(r"^### (.+)$", r"### 💡 \1", response, flags=re.MULTILINE)

    # Add emphasis to important words
    response = re.sub(
        r"\b(important|note|warning|tip)\b",
        r'<span class="highlight-word">\1</span>',
        response,
        flags=re.IGNORECASE,
    )

    return response


def format_chat_history(message_history: List[Message]) -> str:
    """Format chat history for display in the sidebar."""
    if not message_history:
        return "No chat history yet."

    formatted_history = []
    for i, message in enumerate(message_history):
        role = message.role
        content = message.content

        if role == "user":
            formatted_history.append(f"**🧑 User:** {content}")
        elif role == "assistant":
            # Truncate long responses for history display
            truncated_content = content[:200] + "..." if len(content) > 200 else content
            formatted_history.append(f"**🤖 Assistant:** {truncated_content}")

    return "\n\n---\n\n".join(formatted_history)


def send_message(
    client: str, model: str, retrieval: str, system_msg: str, user_msg: str
) -> tuple[str | LLMResponse, str]:
    print(
        f"Sending message with client: {client}, model: {model}, retrieval: {retrieval}, "
        f"system_msg: {system_msg}, user_msg: {user_msg}"
    )
    chatbot.model = model

    print("Instantiating vector db...")
    vector_db = SqliteVectorDB(
        db_path=r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset/bio_vector_db.db",
        embed_fn=embed_text,
    )
    print("Setting vector db...")
    chatbot.vector_db = vector_db
    chatbot.provider = client.lower()
    chatbot.set_client()
    llm_response = chatbot.send_message(
        user_message=user_msg, system_message=system_msg, retrieve_context=retrieval == "On"
    )

    # Enhance the response if it's a string
    if isinstance(llm_response, LLMResponse):
        response = enhance_ai_response(llm_response.content)
    else:
        raise ValueError(f"Invalid response type: {type(llm_response)}")

    # Get formatted chat history from chatbot object
    chat_history = format_chat_history(chatbot.message_history)

    return response, chat_history


with gr.Blocks(
    title="Multi-Client Chatbot Demo",
    css="""
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
    }
    
    #ai-response {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 0px;
        min-height: 200px;
        max-height: 600px;
        overflow-y: auto;
        position: relative;
        align-self: flex-start;
    }
    
    #ai-response:empty:before {
        content: "AI response will appear here...";
        color: #7f8c8d;
        font-style: italic;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 200px;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 6px;
        border: 2px dashed #ddd;
    }
    
    #chat-history {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 15px;
        background: #f8f9fa;
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 15px;
        font-size: 13px;
    }
    
    #chat-history strong {
        color: #2c3e50;
    }
    
    #chat-history hr {
        border: none;
        border-top: 1px solid #dee2e6;
        margin: 10px 0;
    }
    
    #ai-response h1, #ai-response h2, #ai-response h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    #ai-response h1 { font-size: 1.8em; }
    #ai-response h2 { font-size: 1.5em; }
    #ai-response h3 { font-size: 1.3em; }
    
    #ai-response p {
        margin: 12px 0;
        color: #2c3e50;
        font-size: 14px;
    }
    
    #ai-response code {
        background: #f1f2f6;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        color: #e74c3c;
        font-size: 13px;
    }
    
    #ai-response pre {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
        border-left: 4px solid #3498db;
        margin: 15px 0;
    }
    
    #ai-response pre code {
        background: transparent;
        color: #ecf0f1;
        padding: 0;
    }
    
    #ai-response blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        margin: 15px 0;
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        font-style: italic;
    }
    
    #ai-response ul, #ai-response ol {
        padding-left: 25px;
        margin: 12px 0;
    }
    
    #ai-response li {
        margin: 8px 0;
        line-height: 1.5;
    }
    
    #ai-response table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    #ai-response th, #ai-response td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    #ai-response th {
        background: #3498db;
        color: white;
        font-weight: 600;
    }
    
    #ai-response tr:hover {
        background: #f8f9fa;
    }
    
    #ai-response strong {
        color: #2c3e50;
        font-weight: 600;
    }
    
    #ai-response em {
        color: #7f8c8d;
        font-style: italic;
    }
    
    #ai-response a {
        color: #3498db;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-bottom 0.3s;
    }
    
    #ai-response a:hover {
        border-bottom: 1px solid #3498db;
    }
    
         /* Math equation styling */
     #ai-response .katex-display {
         margin: 20px 0;
         background: #f8f9fa;
         padding: 15px;
         border-radius: 8px;
         border: 1px solid #e9ecef;
         text-align: center;
     }
     
     #ai-response .katex {
         font-size: 1.1em;
         color: #2c3e50;
     }
     
     #ai-response .katex-display .katex {
         font-size: 1.2em;
     }
     
     /* Inline math styling */
     #ai-response .katex-inline {
         background: rgba(52, 152, 219, 0.1);
         padding: 2px 4px;
         border-radius: 3px;
     }
    
    /* Loading animation */
    #ai-response.loading::before {
        content: "Thinking...";
        display: block;
        color: #7f8c8d;
        font-style: italic;
        text-align: center;
        padding: 20px;
    }
    

    /* Highlight words */
    .highlight-word {
        background: linear-gradient(120deg, #f39c12 0%, #e74c3c 100%);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        text-transform: uppercase;
    }
    
         /* Custom scrollbar styling */
     #ai-response::-webkit-scrollbar {
         width: 8px;
     }
     
     #ai-response::-webkit-scrollbar-track {
         background: #f1f1f1;
         border-radius: 4px;
     }
     
     #ai-response::-webkit-scrollbar-thumb {
         background: #c1c1c1;
         border-radius: 4px;
         transition: background 0.3s;
     }
     
     #ai-response::-webkit-scrollbar-thumb:hover {
         background: #a1a1a1;
     }
     
     /* Animation for new content */
     #ai-response {
         animation: fadeIn 0.5s ease-in;
     }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
                #ai-response {
            padding: 15px;
            margin: 10px 0;
        }
     }
 """,
    theme=custom_theme,
) as demo:
    gr.Markdown("## AI Chatbot Playground")
    with gr.Row():
        show_sidebar = gr.Checkbox(label="Show chat history (sidebar)", value=True)
        chat_history_box = gr.Markdown(visible=True, value="No chat history yet.", elem_id="chat-history")

    with gr.Row():
        with gr.Column(scale=1):
            client = gr.Dropdown(label="Client", choices=list(CLIENT_MODELS), value="OpenAI")
            model = gr.Dropdown(
                label="Model",
                choices=CLIENT_MODELS["OpenAI"],
                value=CLIENT_MODELS["OpenAI"][0],
            )
            retrieval = gr.Dropdown(label="Retrieval Mode", choices=["Off", "On"], value="Off")
            system_msg = gr.Textbox(
                label="System Message (optional)", placeholder="System message here...", lines=2
            )
            user_msg = gr.Textbox(label="User Message", placeholder="Type your message here...", lines=3)
            send_btn = gr.Button("Send")

        with gr.Column(scale=1):
            ai_output = gr.Markdown(
                label="AI Response",
                value="",
                sanitize_html=False,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
                elem_id="ai-response",
            )

    # -- Logic: update model list when client changes --
    client.change(fn=update_models, inputs=client, outputs=model)
    model.change(fn=select_model, inputs=[client, model], outputs=model)
    retrieval.change(fn=select_retrieval, inputs=[retrieval], outputs=retrieval)
    send_btn.click(
        fn=send_message,
        inputs=[client, model, retrieval, system_msg, user_msg],
        outputs=[ai_output, chat_history_box],
    )

    show_sidebar.change(fn=show_history_toggle, inputs=[show_sidebar], outputs=chat_history_box)

if __name__ == "__main__":
    demo.launch()
