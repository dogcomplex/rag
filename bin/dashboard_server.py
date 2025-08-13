import json, pathlib, threading, time, os, sqlite3, collections, datetime as dt, subprocess, sys
from flask import Flask, send_from_directory, jsonify, request

from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, enqueue

ROOT = pathlib.Path('.knowledge')
BASE_DIR = pathlib.Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'dashboard_static'

app = Flask(__name__, static_folder=str(STATIC_DIR))

from collections import deque
WORKER_PROC = None
WORKER_LOG = deque(maxlen=300)

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
        plugin_status = {}
        for row in cur.execute(
            "select plugin, status, count(*) as n from jobs group by plugin, status"):
            plugin_status.setdefault(row['plugin'], {})[row['status']] = row['n']
        out['by_plugin_status'] = plugin_status
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
    from dotenv import load_dotenv
    import requests
    load_dotenv(override=False)
    base = os.getenv('OPENAI_BASE_URL') or 'http://127.0.0.1:12345/v1'
    try:
        r = requests.get(base.rstrip('/') + '/models', timeout=5)
        ok = r.status_code == 200
        models = r.json().get('data', []) if ok else []
        return {'reachable': ok, 'endpoint': base, 'models': [m.get('id') for m in models[:5]]}
    except Exception:
        return {'reachable': False, 'endpoint': base, 'models': []}

def _read_jobs_status():
    plugins = ['summary-20w','topic-tags','pii-scan','glossary','requirements','todo-items','faq-pairs',
               'keyphrases','bridge-candidates','risk-scan','recent-summary',
               'summary-short','summary-medium','summary-long','summary-outline']
    doc_ids = _unique_doc_ids()
    data = {
        'docs_total': len(doc_ids),
        'domains': _domains_breakdown(),
        'attributes_coverage': _attribute_coverage(plugins, doc_ids),
        'queue': _db_summary(60),
        'llm': _llm_health(),
    }
    return data

@app.get('/api/status')
def api_status():
    data = _read_jobs_status()
    data['worker'] = {
        'running': WORKER_PROC is not None and (WORKER_PROC.poll() is None),
        'log_tail': list(WORKER_LOG)[-50:],
    }
    return jsonify(data)

@app.post('/api/enqueue')
def api_enqueue():
    body = request.get_json(force=True, silent=True) or {}
    plugins = body.get('plugins') or []
    doc_ids = body.get('doc_ids') or []
    payload = body.get('payload') or {}
    cfg = load_configs(); ensure_db(cfg)
    count = 0
    for d in doc_ids:
        for p in plugins:
            enqueue(cfg, p, d, payload)
            count += 1
    return jsonify({'enqueued': count})

@app.post('/api/plan')
def api_plan():
    # spawn plan_enqueue.py as a subprocess to avoid import/package issues
    args = request.get_json(force=True, silent=True) or {}
    cmd = [sys.executable, '-X', 'utf8', 'bin/plan_enqueue.py']
    if 'plugins' in args:
        cmd += ['--plugins', ','.join(args['plugins'])]
    if args.get('only_missing'):
        cmd += ['--only-missing']
    if args.get('limit'):
        cmd += ['--limit', str(args['limit'])]
    if args.get('changed_since_min'):
        cmd += ['--changed-since-min', str(args['changed_since_min'])]
    if args.get('summaries_modes'):
        cmd += ['--summaries-modes', ','.join(args['summaries_modes'])]
    env = os.environ.copy(); env['PYTHONPATH'] = str(pathlib.Path.cwd())
    subprocess.run(cmd, env=env)
    return jsonify({'ok': True})

def _spawn_worker(plugins: list[str], batch: int, watch: bool=True):
    global WORKER_PROC
    if WORKER_PROC and WORKER_PROC.poll() is None:
        return False
    args = [sys.executable, '-X', 'utf8', 'bin/enrich_worker.py', '--plugins', ','.join(plugins), '--batch', str(batch)]
    if watch:
        args.append('--watch')
    env = os.environ.copy(); env['PYTHONPATH'] = str(pathlib.Path.cwd())
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    WORKER_PROC = proc
    def _reader():
        try:
            for line in proc.stdout:
                WORKER_LOG.append(line.rstrip())
        except Exception:
            pass
    threading.Thread(target=_reader, daemon=True).start()
    return True

@app.post('/api/worker/start')
def api_worker_start():
    body = request.get_json(force=True, silent=True) or {}
    plugins = body.get('plugins') or ['summaries','keyphrases','bridge-candidates','risk-scan','recent-summary']
    batch = int(body.get('batch') or 32)
    ok = _spawn_worker(plugins, batch, watch=True)
    return jsonify({'started': ok})

@app.post('/api/worker/stop')
def api_worker_stop():
    global WORKER_PROC
    if WORKER_PROC and (WORKER_PROC.poll() is None):
        try:
            WORKER_PROC.terminate()
        except Exception:
            pass
    WORKER_PROC = None
    return jsonify({'stopped': True})

@app.get('/')
def root():
    index_path = pathlib.Path(app.static_folder) / 'index.html'
    if not index_path.exists():
        return jsonify({'error':'index not found', 'path': str(index_path)}), 500
    return send_from_directory(app.static_folder, 'index.html')

@app.get('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

def run(host='0.0.0.0', port=5051):
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run()

# --------------
# Document + attribute browsing APIs
# --------------

def _first_chunk_for_doc(doc_id: str):
    for p in CHUNKS_DIR.glob(f"{doc_id}-*.json"):
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
    return None

@app.get('/api/docs')
def api_docs():
    docs = []
    for d in sorted(_unique_doc_ids()):
        rec = _first_chunk_for_doc(d) or {}
        meta = rec.get('meta') or {}
        docs.append({'doc_id': d, 'domain': meta.get('domain','root'), 'path': meta.get('path','')})
    return jsonify({'docs': docs})

def _attr_paths_for_doc(doc_id: str):
    out = {}
    if ATTR_DIR.exists():
        for plugin_dir in ATTR_DIR.iterdir():
            if not plugin_dir.is_dir():
                continue
            if plugin_dir.name == 'summaries':
                modes = []
                for f in plugin_dir.glob(f"{doc_id}_*.json"):
                    try:
                        mode = f.stem.split('_', 1)[1]
                        modes.append({'mode': mode, 'path': str(f)})
                    except Exception:
                        continue
                if modes:
                    out['summaries'] = modes
            else:
                f = plugin_dir / f"{doc_id}.json"
                if f.exists():
                    out[plugin_dir.name] = {'path': str(f)}
    return out

@app.get('/api/doc/<doc_id>')
def api_doc(doc_id):
    rec = _first_chunk_for_doc(doc_id) or {}
    meta = rec.get('meta') or {}
    attrs = _attr_paths_for_doc(doc_id)
    return jsonify({'doc_id': doc_id, 'meta': meta, 'attributes': attrs})

@app.get('/api/doc/<doc_id>/attr/<plugin>')
def api_doc_attr(doc_id, plugin):
    if plugin == 'summaries':
        pdir = ATTR_DIR / 'summaries'
        items = []
        for f in sorted(pdir.glob(f"{doc_id}_*.json")):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                items.append(data)
            except Exception:
                continue
        return jsonify({'items': items})
    p = ATTR_DIR / plugin / f"{doc_id}.json"
    if not p.exists():
        return jsonify({'error':'not found'}), 404
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        data = {'raw': p.read_text(errors='ignore')}
    return jsonify({'item': data})

# Serve existing report (if present) under /report
REPORT_DIR = ROOT / 'exports' / 'reports'

@app.get('/report')
def report_index():
    idx = REPORT_DIR / 'index.html'
    if not idx.exists():
        return jsonify({'error': 'report not found', 'path': str(idx)}), 404
    return send_from_directory(str(REPORT_DIR), 'index.html')

@app.get('/report/<path:path>')
def report_static(path):
    if not REPORT_DIR.exists():
        return jsonify({'error': 'report dir not found'}), 404
    return send_from_directory(str(REPORT_DIR), path)

