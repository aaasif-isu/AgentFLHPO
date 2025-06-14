
# In code/agent/llm_api.py
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = "http://localhost:3000"
YOUR_APP_NAME = "FedHPO"

def call_llm(prompt: str) -> str:
    # --- ADD THIS BLOCK TO PRINT THE PROMPT FOR DEBUGGING ---
    # print("\n" + "="*60)
    # print(">>> PROMPT BEING SENT TO LLM API:")
    # print(prompt)
    # print("="*60 + "\n")
    # --- END DEBUGGING BLOCK ---

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content

    except requests.exceptions.HTTPError as http_err:
        print(f"API request error: {http_err}")
        return "" # Return empty string on HTTP error
    except Exception as e:
        print(f"An unexpected error occurred in call_llm: {e}")
        return "" # Return empty string on other errors