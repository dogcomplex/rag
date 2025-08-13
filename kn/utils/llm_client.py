import os, requests
from dotenv import load_dotenv

# Load .env, but don't override a process-level env already set
load_dotenv(override=False)

def _settings():
    base = os.getenv('OPENAI_BASE_URL') or 'http://127.0.0.1:12345/v1'
    key  = os.getenv('OPENAI_API_KEY', 'lm-studio')
    model= os.getenv('OPENAI_MODEL', 'llama-3.1-8b-instruct')
    return base, key, model

def chat(prompt: str, max_tokens=512, temperature=0.2):
    BASE, KEY, MODEL = _settings()
    url = f"{BASE}/chat/completions"
    headers = {'Authorization': f"Bearer {KEY}", 'Content-Type':'application/json'}
    payload = {
        'model': MODEL,
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']