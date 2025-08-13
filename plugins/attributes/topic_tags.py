import sys, json, pathlib
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text
OUTDIR = pathlib.Path('.knowledge/indexes/attributes/topic-tags'); OUTDIR.mkdir(parents=True, exist_ok=True)
for line in sys.stdin:
    job = json.loads(line); text = load_skeleton_text(job.get('doc_id')) or job.get('text','')
    if not text: continue
    prompt = ("Extract 3-7 topical tags (comma-separated, lowercase, no spaces, use-hyphens).\n"
              "Prefer domain-relevant terms.\n\n"
              f"TEXT:\n{text}\n")
    out = chat(prompt, max_tokens=64, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        "doc_id": job['doc_id'], "attribute": "topic-tags", "value": out.strip(), "confidence": 0.7, "pass":"cheap"
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"status":"ok","doc_id":job['doc_id']}))