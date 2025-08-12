import sys, json, pathlib
from kn.utils.llm_client import chat
OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-20w'); OUTDIR.mkdir(parents=True, exist_ok=True)
for line in sys.stdin:
    job = json.loads(line); text = job.get('text','')
    if not text: continue
    prompt = f"Summarize in ~20 words, terse, factual.\n\n{text}"
    out = chat(prompt, max_tokens=100, temperature=0.1)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        "doc_id": job['doc_id'], "attribute": "summary-20w", "value": out.strip(), "confidence": 0.8, "pass": "cheap"
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"status":"ok","doc_id":job['doc_id']}))