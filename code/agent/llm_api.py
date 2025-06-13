
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


# import requests
# import json
# import os
# from dotenv import load_dotenv

# def call_llm(prompt: str) -> str:
#     """
#     Calls OpenRouter API with the specified model and prompt.
#     Args:
#         prompt: Input prompt for the LLM.
#     Returns:
#         JSON string, e.g., {"response": [...]}
#     """
#     load_dotenv()
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         print("Error: OPENROUTER_API_KEY not found in .env file.")
#         return json.dumps({"response": []})

#     model = "openai/gpt-4o-mini"
#     #openai/gpt-4o-mini
#     #deepseek/deepseek-r1-0528:free
#     #deepseek/deepseek-v3-base:free
#     #google/gemma-3-4b-it:free
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant. Return JSON without code blocks."},
#             {"role": "user", "content": prompt}
#         ],
#         "response_format": {"type": "json_object"}
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload, timeout=30)
#         response.raise_for_status()
#         result = response.json()
#         content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

#         # --- NEW PRINT STATEMENT ---
#         # print("\n--- [RAW LLM RESPONSE] ---")
#         # print(content)
#         # print("--- [END RAW LLM RESPONSE] ---\n")
#         # --- END OF NEW PRINT STATEMENT ---

#         # Strip Markdown code blocks and whitespace
#         content = content.strip()
#         if content.startswith("```json"):
#             content = content[7:].rstrip("```").strip()
#         try:
#             parsed = json.loads(content)
#             if not isinstance(parsed, dict):
#                 parsed = {"response": parsed}
#             elif "response" not in parsed:
#                 parsed = {"response": parsed}
#             return json.dumps(parsed)
#         except json.JSONDecodeError as e:
#             print(f"Error: Malformed JSON in API response: {e}, content: {content}")
#             return json.dumps({"response": []})
#     except requests.RequestException as e:
#         print(f"API request error: {e}")
#         return json.dumps({"response": []})