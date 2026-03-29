import os
import json
import requests
import subprocess
from urllib.parse import quote

# ---------------------------------------------------------
# Ollama Configuration
# ---------------------------------------------------------
# Default endpoint for local Ollama. Ensure your Ollama server is running!
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# Using llama3 as the default agent model. You can change this to mistral, qwen, etc.
MODEL_NAME = "llama3.1"

# ---------------------------------------------------------
# Tool Definitions (Agent Reach style)
# ---------------------------------------------------------

def read_webpage_markdown(url: str) -> str:
    """Uses Jina Reader to fetch clean markdown of a webpage, simulating Agent Reach."""
    print(f"  [Tool Executing] -> r.jina.ai/{url}")
    try:
        # Agent Reach typically uses simple curl for Jina
        result = subprocess.run(
            ["curl", "-s", "--", f"https://r.jina.ai/{url}"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        return f"Error executing jina reader tool: {str(e)}"

def fetch_raw_html(url: str) -> str:
    """Fetches raw HTML directly via curl."""
    print(f"  [Tool Executing] -> direct curl {url}")
    try:
        result = subprocess.run(
            ["curl", "-s", "-A", "Mozilla/5.0", "--", url],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        return f"Error executing raw curl tool: {str(e)}"

# The JSON schema describing our tools to the LLaMA model
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_webpage_markdown",
            "description": "Reads a webpage and returns its content in Markdown format. Excellent for reading articles or clean text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_raw_html",
            "description": "Fetches raw HTML directly from a URL. Best if you need the raw source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch HTML from."
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# Map tool names to actual Python functions
AVAILABLE_TOOLS = {
    "read_webpage_markdown": read_webpage_markdown,
    "fetch_raw_html": fetch_raw_html
}

# ---------------------------------------------------------
# Ollama Agent Logic
# ---------------------------------------------------------

def chat_with_ollama(messages, tools):
    """Sends a chat request to Ollama with tool schemas attached."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"\n[Error] Cannot connect to Ollama at {OLLAMA_API_URL}.")
        print("Please ensure the Ollama application is running and the model is pulled.")
        print(f"Command to pull: 'ollama pull {MODEL_NAME}'")
        return None
    except Exception as e:
        print(f"\n[Error] Ollama API error: {e}")
        return None

def process_keyword_with_agent(keyword: str, targets_dir: str):
    """Runs the full agentic loop for a single keyword."""
    print(f"\n======================================")
    print(f"Agent starting task for: '{keyword}'")

    query = quote(keyword)
    target_url = f"https://www.trendyol.com/sr?q={query}"

    # 1. System prompt instructing the agent what to do
    system_prompt = (
        "You are an autonomous web scraping agent equipped with tools to read webpages. "
        "Your task is to fetch the search results for a specific e-commerce keyword on Trendyol. "
        "You must use a tool to fetch the content of the provided URL. Because we need the raw HTML for our legacy extractor, "
        "you should prioritize fetching raw HTML, but you can try reading markdown if that fails."
    )

    user_prompt = f"Please fetch the search results for the URL: {target_url}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 2. Ask Ollama what to do
    print("[Agent] Thinking...")
    response_data = chat_with_ollama(messages, TOOLS_SCHEMA)

    if not response_data:
        print(f"[Agent] Failed to get a response for '{keyword}'. Skipping.")
        return

    message = response_data.get("message", {})

    # 3. Check if the LLM decided to call a tool
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            function_call = tool_call.get("function", {})
            tool_name = function_call.get("name")

            # Ensure arguments are parsed (Ollama returns them as a dict)
            tool_args = function_call.get("arguments", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except:
                    tool_args = {}

            print(f"[Agent] Decided to use tool: {tool_name}")
            print(f"[Agent] Tool arguments: {tool_args}")

            # Execute the actual Python tool function
            if tool_name in AVAILABLE_TOOLS:
                tool_function = AVAILABLE_TOOLS[tool_name]
                url_to_fetch = tool_args.get("url", target_url)

                # RUN THE TOOL!
                content = tool_function(url_to_fetch)

                print(f"[Agent] Tool executed successfully. Received {len(content)} bytes of data.")

                # Check for anti-bot blocks
                if "cf-browser-verification" in content or len(content) < 5000:
                    print("[Agent] WARNING: The site returned very little data or a Cloudflare challenge.")
                    print("[Agent] The standard curl/Jina tools cannot execute Javascript infinite scrolling.")

                # Save the output to targets/
                safe_keyword = keyword.replace(" ", "-")
                output_file = os.path.join(targets_dir, f"{safe_keyword}-target.html")

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"[Agent] Saved extracted payload to {output_file}")

                # We stop after the first successful tool execution for this simple pipeline
                break
            else:
                print(f"[Agent] Tried to call unknown tool: {tool_name}")
    else:
        # The LLM just replied with text instead of calling a tool
        print("[Agent] Did not call any tools. Response:")
        print(message.get("content"))

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def scrape_trendyol_ollama(keywords_file="keywords.txt", targets_dir="../targets"):
    if not os.path.exists(targets_dir):
        os.makedirs(targets_dir)

    if not os.path.exists(keywords_file):
        print(f"Error: {keywords_file} not found.")
        return

    with open(keywords_file, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        print("No keywords found.")
        return

    print(f"Starting Ollama Agentic Scraper for {len(keywords)} keywords...")
    print(f"Connecting to Ollama at {OLLAMA_API_URL} using model '{MODEL_NAME}'")

    for keyword in keywords:
        process_keyword_with_agent(keyword, targets_dir)

    print("\nOllama Agentic scraping completed.")

if __name__ == "__main__":
    scrape_trendyol_ollama()
