import os, requests
BASE = os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1')
KEY  = os.getenv('OPENAI_API_KEY', 'lm-studio')
MODEL= os.getenv('OPENAI_MODEL', 'llama-3.1-8b-instruct')
def chat(prompt: str, max_tokens=512, temperature=0.2):
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