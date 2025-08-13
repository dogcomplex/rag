import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summaries')
OUTDIR.mkdir(parents=True, exist_ok=True)

TEMPLATES = {
    'short':   ("Summarize the text in ~50 words, terse, factual.", 120),
    'medium':  ("Summarize the text in ~150 words, clear sections if relevant.", 360),
    'long':    ("Summarize the text in ~400 words, comprehensive but concise.", 900),
    'outline': ("Produce a hierarchical outline (bullets) of the main ideas.", 360),
}

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    payload = job.get('payload') or {}
    mode = (payload.get('mode') or 'short').lower()
    words = payload.get('words')
    tpl, max_tokens = TEMPLATES.get(mode, TEMPLATES['short'])
    if words:
        tpl = f"Summarize the text in ~{words} words, terse and factual."
        max_tokens = int(words) * 3
    prompt = tpl + "\n\n" + text
    out = chat(prompt, max_tokens=max_tokens, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}_{mode}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': f'summary-{mode}', 'value': out.strip(), 'mode': mode, 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id'],'mode':mode}))

