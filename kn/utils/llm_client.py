import os, requests
from dotenv import load_dotenv
from .cache import get_cached_response, set_cached_response
from kn.config import load_configs

# Load .env, but don't override a process-level env already set
load_dotenv(override=False)

def _settings(overrides: dict|None=None):
    base = (overrides or {}).get('base_url') or os.getenv('OPENAI_BASE_URL') or 'http://127.0.0.1:12345/v1'
    key  = (overrides or {}).get('api_key') or os.getenv('OPENAI_API_KEY', 'lm-studio')
    model= (overrides or {}).get('model') or os.getenv('OPENAI_MODEL', 'llama-3.1-8b-instruct')
    timeout = (overrides or {}).get('timeout') or 120
    return base, key, model, timeout

def _plugin_overrides(plugin_name: str|None) -> dict:
    if not plugin_name:
        return {}
    try:
        cfg = load_configs()
        plug = (cfg.get('plugins') or {}).get(plugin_name) or {}
        llm = plug.get('llm') or {}
        return llm
    except Exception:
        return {}

def chat(prompt: str, max_tokens=512, temperature=0.2, overrides: dict|None=None, cache_key: str|None=None, plugin_name: str|None=None):
    merged = {}
    # config defaults per plugin
    cfg_over = _plugin_overrides(plugin_name)
    merged.update(cfg_over)
    # runtime overrides take precedence
    if overrides:
        merged.update(overrides)
    BASE, KEY, MODEL, TIMEOUT = _settings(merged)
    max_tokens = merged.get('max_tokens', max_tokens)
    temperature = merged.get('temperature', temperature)
    top_p = merged.get('top_p')
    url = f"{BASE}/chat/completions"
    headers = {'Authorization': f"Bearer {KEY}", 'Content-Type':'application/json'}
    payload = {
        'model': MODEL,
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    if top_p is not None:
        payload['top_p'] = top_p
    # optional local model directory hint (vLLM/OAI adapters may ignore it)
    local_models = os.getenv('LOCAL_MODELS_DIR')
    if local_models:
        payload['extra_body'] = {'local_models_dir': local_models}
    # cache key includes model + prompt + max_tokens + temperature for stability
    ck = cache_key or f"{MODEL}|{max_tokens}|{temperature}|{prompt}"
    cached = get_cached_response(ck)
    if cached is not None:
        return cached
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    content = r.json()['choices'][0]['message']['content']
    set_cached_response(ck, content)
    return content