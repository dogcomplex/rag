import argparse, json, pathlib, collections
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, enqueue
from kn.utils.llm_client import chat

CHUNK_DIR = pathlib.Path('.knowledge/indexes/chunks')
COMM_PATH = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')

def collect_docs_and_domains():
    doc_to_domain: dict[str, str] = {}
    domain_counts = collections.Counter()
    for p in CHUNK_DIR.glob('*.json'):
        try:
            rec = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        doc_id = rec.get('doc_id')
        meta = rec.get('meta', {})
        domain = meta.get('domain', 'root') or 'root'
        if doc_id not in doc_to_domain:
            doc_to_domain[doc_id] = domain
            domain_counts[domain] += 1
    return doc_to_domain, domain_counts

def load_community_summaries(limit=8):
    out = []
    if COMM_PATH.exists():
        for line in COMM_PATH.read_text(encoding='utf-8').splitlines()[:limit]:
            try:
                c = json.loads(line)
                out.append(f"{c.get('id')} size={c.get('size')}: {c.get('summary')}")
            except Exception:
                continue
    return out

SYS_PROMPT = (
    "You are a planning agent for repository analysis. Based on the context, propose high-value attribute jobs.\n"
    "Available plugins: ['summary-20w','topic-tags','pii-scan','glossary','requirements','todo-items','faq-pairs'].\n"
    "Scopes you may use: 'all' or a list of domains from the provided list.\n"
    "Return STRICT JSON: {\"jobs\": [{\"plugin\": str, \"scope\": str, \"limit\": int|null}...]}.\n"
    "Choose at most 6 jobs. Prefer targeted scopes over 'all'."
)

def build_planner_context():
    doc_to_domain, domain_counts = collect_docs_and_domains()
    domains_list = ', '.join([f"{d}({n})" for d, n in domain_counts.most_common()]) or 'root(0)'
    comms = load_community_summaries()
    ctx = [
        "Context:",
        f"Domains: {domains_list}",
        "Top communities:",
    ] + [f"- {c}" for c in comms]
    return "\n".join(ctx), doc_to_domain

def expand_scope(scope: str, doc_to_domain: dict[str,str]) -> list[str]:
    if scope.strip().lower() == 'all':
        return list(doc_to_domain.keys())
    if scope.startswith('domain:'):
        want = scope.split(':',1)[1].strip().lower()
        return [doc for doc, dom in doc_to_domain.items() if (dom or '').lower() == want]
    # Unknown scope → no docs
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-jobs', type=int, default=64)
    args = ap.parse_args()

    cfg = load_configs()
    ensure_db(cfg)
    context, doc_to_domain = build_planner_context()

    prompt = SYS_PROMPT + "\n\n" + context + "\n\nReturn JSON now."
    try:
        raw = chat(prompt, max_tokens=400, temperature=0.2)
        plan = json.loads(raw)
        jobs = plan.get('jobs', []) if isinstance(plan, dict) else []
    except Exception:
        jobs = []

    enq = 0
    for job in jobs:
        plugin = str(job.get('plugin','')).strip()
        scope = str(job.get('scope','all')).strip()
        limit = job.get('limit')
        docs = expand_scope(scope, doc_to_domain)
        if limit and isinstance(limit, int):
            docs = docs[:max(0, limit)]
        for doc_id in docs:
            enqueue(cfg, plugin, doc_id, payload={})
            enq += 1
            if enq >= args.max_jobs:
                break
        if enq >= args.max_jobs:
            break
    print(f"[planner] enqueued {enq} jobs from LLM plan")

if __name__ == '__main__':
    main()

