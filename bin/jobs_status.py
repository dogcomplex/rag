import argparse
import collections
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys

from dotenv import load_dotenv
import requests


CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')
ATTR_DIR = pathlib.Path('.knowledge/indexes/attributes')
DB_PATH = pathlib.Path('.knowledge/queues/jobs.sqlite')


def _unique_doc_ids():
    seen = set()
    if not CHUNKS_DIR.exists():
        return seen
    for p in CHUNKS_DIR.glob('*.json'):
        try:
            doc_id = p.stem.split('-')[0]
            seen.add(doc_id)
        except Exception:
            continue
    return seen


def _attribute_coverage(plugins, doc_ids):
    coverage = {}
    for plugin in plugins:
        p = ATTR_DIR / plugin
        have = 0
        missing = []
        if p.exists():
            existing = {f.stem for f in p.glob('*.json')}
            for d in doc_ids:
                if d in existing:
                    have += 1
                else:
                    missing.append(d)
        else:
            missing = list(doc_ids)
        total = len(doc_ids)
        coverage[plugin] = {
            'total_docs': total,
            'have': have,
            'pct': round(100.0 * (have / total), 1) if total else 0.0,
            'missing_examples': missing[:5],
        }
    return coverage


def _domains_breakdown():
    from json import loads
    dom = collections.Counter()
    for p in CHUNKS_DIR.glob('*.json'):
        try:
            rec = loads(p.read_text(encoding='utf-8'))
            d = (rec.get('meta') or {}).get('domain') or 'root'
            doc_id = rec.get('doc_id')
            # count per-doc by domain (first chunk only)
            dom[(doc_id, d)] += 1
        except Exception:
            continue
    out = collections.Counter()
    for (_, d), _ in dom.items():
        out[d] += 1
    return [{'domain': k, 'docs': v} for k, v in out.most_common()]


def _db_summary(minutes_recent=60):
    if not DB_PATH.exists():
        return {
            'present': False,
            'total': 0,
            'by_status': {},
            'by_plugin_status': {},
            'oldest_pending_min': None,
            'recent_created': {},
        }
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    out = {'present': True}
    try:
        by_status = {k: v for k, v in cur.execute(
            "select status, count(*) from jobs group by status").fetchall()}
        out['by_status'] = by_status
        total = sum(by_status.values()) if by_status else 0
        out['total'] = total

        # by plugin, status
        plugin_status = {}
        for row in cur.execute(
            "select plugin, status, count(*) as n from jobs group by plugin, status"):
            plugin_status.setdefault(row['plugin'], {})[row['status']] = row['n']
        out['by_plugin_status'] = plugin_status

        # oldest pending age (minutes)
        row = cur.execute(
            "select created_at from jobs where status='pending' order by id asc limit 1").fetchone()
        if row and row['created_at']:
            try:
                created = dt.datetime.fromisoformat(str(row['created_at']))
                age_min = (dt.datetime.now() - created).total_seconds() / 60.0
            except Exception:
                age_min = None
        else:
            age_min = None
        out['oldest_pending_min'] = round(age_min, 1) if age_min is not None else None

        # recent created by plugin within window
        recent = {}
        try:
            cutoff = dt.datetime.now() - dt.timedelta(minutes=minutes_recent)
            cutoff_iso = cutoff.isoformat(sep=' ')
            for row in cur.execute(
                "select plugin, count(*) as n from jobs where created_at >= ? group by plugin",
                (cutoff_iso,)):
                recent[row['plugin']] = row['n']
        except Exception:
            recent = {}
        out['recent_created'] = recent
    finally:
        con.close()
    return out


def _llm_health():
    load_dotenv(override=False)
    base = os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1')
    try:
        r = requests.get(base.rstrip('/') + '/models', timeout=5)
        ok = r.status_code == 200
        models = r.json().get('data', []) if ok else []
        return {'reachable': ok, 'endpoint': base, 'models': [m.get('id') for m in models[:5]]}
    except Exception:
        return {'reachable': False, 'endpoint': base, 'models': []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recent-mins', type=int, default=60)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    plugins = ['summary-20w','topic-tags','pii-scan','glossary','requirements','todo-items','faq-pairs']
    doc_ids = _unique_doc_ids()
    data = {
        'docs_total': len(doc_ids),
        'domains': _domains_breakdown(),
        'attributes_coverage': _attribute_coverage(plugins, doc_ids),
        'queue': _db_summary(args.recent_mins),
        'llm': _llm_health(),
    }

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # Pretty print
    print(f"docs_total: {data['docs_total']}")
    print("domains:")
    for d in data['domains'][:10]:
        print(f"  - {d['domain']}: {d['docs']}")
    print("attributes coverage:")
    for k, v in data['attributes_coverage'].items():
        print(f"  - {k}: {v['have']}/{v['total_docs']} ({v['pct']}%)")
    print("queue:")
    q = data['queue']
    print(f"  present: {q['present']}")
    if q['present']:
        print(f"  total: {q.get('total',0)}  by_status: {q.get('by_status',{})}")
        print(f"  oldest_pending_min: {q.get('oldest_pending_min')}")
        if q.get('recent_created'):
            print(f"  recent_created({args.recent_mins}m): {q['recent_created']}")
        print("  by_plugin_status:")
        for plug, m in q.get('by_plugin_status',{}).items():
            print(f"    - {plug}: {m}")
    print("llm:")
    llm = data['llm']
    print(f"  reachable: {llm['reachable']}  endpoint: {llm['endpoint']}")
    if llm['models']:
        print(f"  models: {', '.join(llm['models'])}")


if __name__ == '__main__':
    main()

