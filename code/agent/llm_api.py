import requests
import json
import os
from dotenv import load_dotenv

def call_llm(prompt: str) -> str:
    """
    Calls OpenRouter API with the specified model and prompt.
    Args:
        prompt: Input prompt for the LLM.
    Returns:
        JSON string, e.g., {"response": [...]}
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
        return json.dumps({"response": []})

    model = "openai/gpt-4o-mini"
    #openai/gpt-4o-mini
    #deepseek/deepseek-r1-0528:free
    #deepseek/deepseek-v3-base:free
    #google/gemma-3-4b-it:free
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. Return JSON without code blocks."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # --- NEW PRINT STATEMENT ---
        # print("\n--- [RAW LLM RESPONSE] ---")
        # print(content)
        # print("--- [END RAW LLM RESPONSE] ---\n")
        # --- END OF NEW PRINT STATEMENT ---

        # Strip Markdown code blocks and whitespace
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].rstrip("```").strip()
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                parsed = {"response": parsed}
            elif "response" not in parsed:
                parsed = {"response": parsed}
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            print(f"Error: Malformed JSON in API response: {e}, content: {content}")
            return json.dumps({"response": []})
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return json.dumps({"response": []})