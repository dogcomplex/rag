import sys, json, pathlib
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text

OUT = pathlib.Path('.knowledge/indexes/attributes')
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "You are given a document. Produce a compact JSON with keys:"
    " summary_short (~50w), summary_medium (~150w), outline (bulleted), tags (3-7 comma-separated), risk (JSON with contradictions/speculation/outdated/pii_hint boolean and notes<=40w).\n"
    "Return only JSON.\n\nTEXT:\n{body}\n"
)

def write_attr(attr, doc_id, value, extra=None):
    (OUT/attr).mkdir(parents=True, exist_ok=True)
    rec = {'doc_id': doc_id, 'attribute': attr, 'value': value}
    if isinstance(extra, dict): rec.update(extra)
    (OUT/attr/f"{doc_id}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')

for line in sys.stdin:
    job = json.loads(line)
    text = load_skeleton_text(job.get('doc_id')) or job.get('text','') or ''
    if not text.strip():
        continue
    doc_id = job.get('doc_id')
    # Allow per-plugin model override via payload
    overrides = (job.get('payload') or {}).get('llm') or {}
    # keep under tighter limits to reduce timeouts
    raw = chat(PROMPT.format(body=text), max_tokens=400, temperature=0.2, overrides=overrides, plugin_name='multi-basic', overwrite=True)
    data = {}
    try:
        data = json.loads(raw)
    except Exception:
        # best-effort parse: write whole text as medium summary
        data = {'summary_medium': raw.strip()}
    # write individual attribute files
    if 'summary_short' in data:
        write_attr('summary-short', doc_id, data['summary_short'], {'confidence':0.7})
    if 'summary_medium' in data:
        write_attr('summary-medium', doc_id, data['summary_medium'], {'confidence':0.7})
    if 'outline' in data:
        write_attr('summary-outline', doc_id, data['outline'], {'confidence':0.7})
    tags = data.get('tags')
    if isinstance(tags, str):
        write_attr('keyphrases', doc_id, tags, {'confidence':0.7})
    risk = data.get('risk')
    if isinstance(risk, dict):
        write_attr('risk-scan', doc_id, risk)
    print(json.dumps({'status':'ok','doc_id':doc_id}))

