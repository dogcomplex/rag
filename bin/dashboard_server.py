import json, pathlib, threading, time, os, sqlite3, collections, datetime as dt, subprocess, sys
import yaml
from flask import Flask, send_from_directory, jsonify, request

from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, enqueue

ROOT = pathlib.Path('.knowledge')
BASE_DIR = pathlib.Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'dashboard_static'

app = Flask(__name__, static_folder=str(STATIC_DIR))


from collections import deque
import itertools, datetime
_WORKER_ID_GEN = itertools.count(1)
WORKERS: dict[int, dict] = {}

CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')
ATTR_DIR = pathlib.Path('.knowledge/indexes/attributes')
DB_PATH = pathlib.Path('.knowledge/queues/jobs.sqlite')


def _handle_worker_line(wid: int, line: str):
    w = WORKERS.get(wid)
    if not w:
        return
    if line.startswith('[worker-current]'):
        # expected format: [worker-current] plugin=foo docs=doc1,doc2,…
        try:
            rest = line.split(']', 1)[1].strip()
            parts = {}
            for part in rest.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    parts[k] = v
            w['current'] = {
                'plugin': parts.get('plugin'),
                'docs': parts.get('docs'),
                'since': datetime.datetime.now().isoformat(timespec='seconds'),
            }
        except Exception:
            w['current'] = {'raw': line}
    elif line.startswith('[worker-current-clear]'):
        w['current'] = None

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

LLM_MODELS_CACHE = {'data': None, 'ts': 0}

def _llm_health(force: bool=False, ttl_sec: int=600):
    from dotenv import load_dotenv
    import requests
    load_dotenv(override=False)
    base = os.getenv('OPENAI_BASE_URL') or 'http://127.0.0.1:12345/v1'
    now = time.time()
    if not force and LLM_MODELS_CACHE['data'] and (now - LLM_MODELS_CACHE['ts'] < ttl_sec):
        d = dict(LLM_MODELS_CACHE['data'])
        d['endpoint'] = base
        return d
    try:
        r = requests.get(base.rstrip('/') + '/models', timeout=5)
        ok = r.status_code == 200
        models = r.json().get('data', []) if ok else []
        data = {'reachable': ok, 'endpoint': base, 'models': [m.get('id') for m in models[:5]]}
        LLM_MODELS_CACHE['data'] = data
        LLM_MODELS_CACHE['ts'] = now
        return data
    except Exception:
        data = {'reachable': False, 'endpoint': base, 'models': []}
        LLM_MODELS_CACHE['data'] = data
        LLM_MODELS_CACHE['ts'] = now
        return data

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
    if args.get('map_reduce'):
        cmd += ['--map-reduce']
    env = os.environ.copy(); env['PYTHONPATH'] = str(pathlib.Path.cwd())
    subprocess.run(cmd, env=env)
    return jsonify({'ok': True})

def _spawn_worker(plugins: list[str], batch: int, watch: bool=True):
    args = [sys.executable, '-X', 'utf8', 'bin/enrich_worker.py', '--plugins', ','.join(plugins), '--batch', str(batch)]
    if watch:
        args.append('--watch')
    env = os.environ.copy(); env['PYTHONPATH'] = str(pathlib.Path.cwd())
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    wid = next(_WORKER_ID_GEN)
    wlog = deque(maxlen=500)
    WORKERS[wid] = {
        'proc': proc,
        'plugins': plugins,
        'batch': batch,
        'watch': watch,
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'log': wlog,
        'current': None,
    }
    def _reader():
        try:
            for line in proc.stdout:
                clean = line.rstrip()
                wlog.append(clean)
                _handle_worker_line(wid, clean)
        except Exception:
            pass
    threading.Thread(target=_reader, daemon=True).start()
    return wid

@app.post('/api/worker/start')
def api_worker_start():
    body = request.get_json(force=True, silent=True) or {}
    plugins = body.get('plugins') or ['summaries','keyphrases','bridge-candidates','risk-scan','recent-summary']
    batch = int(body.get('batch') or 32)
    wid = _spawn_worker(plugins, batch, watch=True)
    return jsonify({'started': True, 'worker_id': wid})

@app.post('/api/worker/stop')
def api_worker_stop():
    body = request.get_json(force=True, silent=True) or {}
    wid = body.get('id') or body.get('worker_id')
    if wid is None:
        # stop all
        for wid2 in list(WORKERS.keys()):
            _stop_worker_id(wid2)
        return jsonify({'stopped_all': True})
    ok = _stop_worker_id(int(wid))
    return jsonify({'stopped': ok, 'worker_id': wid})

@app.get('/api/workers')
def api_workers():
    out = []
    for wid, w in WORKERS.items():
        proc = w.get('proc')
        out.append({
            'id': wid,
            'pid': proc.pid if proc else None,
            'running': (proc is not None and proc.poll() is None),
            'plugins': w.get('plugins'),
            'batch': w.get('batch'),
            'watch': w.get('watch'),
            'started_at': w.get('started_at'),
        })
    return jsonify({'workers': out})

@app.get('/api/queue/list')
def api_queue_list():
    try:
        limit = int(request.args.get('limit', '200'))
    except Exception:
        limit = 200
    status = request.args.get('status')  # optional: pending|running|done
    plugin = request.args.get('plugin')  # optional
    if not DB_PATH.exists():
        return jsonify({'items': []})
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    q = "select id, plugin, doc_id, status, created_at from jobs"
    cond = []
    args = []
    if status:
        cond.append("status=?"); args.append(status)
    if plugin:
        cond.append("plugin=?"); args.append(plugin)
    if cond:
        q += " where " + " and ".join(cond)
    q += " order by id asc limit ?"; args.append(limit)
    rows = [dict(r) for r in cur.execute(q, args).fetchall()]
    con.close()
    return jsonify({'items': rows})

@app.post('/api/queue/clear')
def api_queue_clear():
    mode = (request.get_json(force=True, silent=True) or {}).get('mode') or 'non-done'
    app.logger.info('[queue-clear] mode=%s', mode)
    if not DB_PATH.exists():
        return jsonify({'ok': True, 'cleared': 0, 'mode': mode})
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cleared = 0
    try:
        if mode == 'all':
            cur.execute('delete from jobs')
            cleared = cur.rowcount
        elif mode == 'reset-running':
            cur.execute("update jobs set status='pending' where status='running'")
            cleared = cur.rowcount
        elif mode == 'pending':
            cur.execute("delete from jobs where status='pending'")
            cleared = cur.rowcount
        elif mode == 'non-done':
            cur.execute("delete from jobs where status!='done'")
            cleared = cur.rowcount
        else:
            return jsonify({'error': 'unknown mode', 'mode': mode}), 400
        con.commit()
    finally:
        con.close()
    return jsonify({'ok': True, 'cleared': int(cleared), 'mode': mode})


# Plugin defaults (persisted to .knowledge/config/models.yml under 'plugins')

# Cache management
CACHE_DIR = pathlib.Path('.knowledge/cache/llm')

@app.get('/api/cache/llm/stats')
def api_cache_stats():
    total = 0; count = 0
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob('*.json'):
            try:
                total += p.stat().st_size
                count += 1
            except Exception:
                continue
    return jsonify({'count': count, 'total_bytes': total})

@app.get('/api/cache/llm/list')
def api_cache_list():
    try:
        limit = int(request.args.get('limit', '200'))
    except Exception:
        limit = 200
    items = []
    if CACHE_DIR.exists():
        files = sorted(CACHE_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:limit]:
            try:
                st = p.stat()
                items.append({'name': p.name, 'size': st.st_size, 'mtime': dt.datetime.fromtimestamp(st.st_mtime).isoformat(sep=' ')})
            except Exception:
                continue
    return jsonify({'items': items})

@app.post('/api/cache/llm/clear')
def api_cache_clear():
    cleared = 0
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob('*.json'):
            try:
                p.unlink(); cleared += 1
            except Exception:
                continue
    return jsonify({'ok': True, 'cleared': cleared})


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

@app.post('/api/ingest')
def api_ingest():
    body = request.get_json(force=True, silent=True) or {}
    repo = body.get('repo') or body.get('path')
    full = bool(body.get('full', True))
    if not repo:
        return jsonify({'error':'missing repo path'}), 400
    # spawn ingest_build_graph.py
    env = os.environ.copy(); env['PYTHONPATH'] = str(pathlib.Path.cwd())
    args = [sys.executable, '-X', 'utf8', 'bin/ingest_build_graph.py', '--repo', repo]
    if full:
        args.append('--full')
    subprocess.Popen(args, env=env)
    return jsonify({'ok': True, 'started': True, 'repo': repo, 'full': full})

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

# Chunk-level attributes per document
def _list_chunk_attrs_for_doc(doc_id: str, plugin: str):
    items = []
    pdir = ATTR_DIR / plugin
    if not pdir.exists():
        return []
    for f in sorted(pdir.glob('*.json')):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('doc_id') != doc_id:
            continue
        chunk_id = data.get('chunk_id')
        seq = None
        try:
            if chunk_id:
                cpath = CHUNKS_DIR / f"{chunk_id}.json"
                if cpath.exists():
                    crec = json.loads(cpath.read_text(encoding='utf-8'))
                    seq = ((crec.get('meta') or {}).get('seq'))
        except Exception:
            pass
        val = data.get('value')
        preview = None
        if isinstance(val, str):
            preview = val.strip().replace('\n',' ')[:200]
        items.append({'chunk_id': chunk_id, 'seq': seq, 'path': str(f), 'preview': preview})
    # order by seq when available, else by chunk_id
    def _key(it):
        if isinstance(it.get('seq'), int):
            return (0, it['seq'])
        return (1, str(it.get('chunk_id') or ''))
    items.sort(key=_key)
    return items

@app.get('/api/doc/<doc_id>/chunks/attr/<plugin>')
def api_doc_chunk_attrs(doc_id, plugin):
    items = _list_chunk_attrs_for_doc(doc_id, plugin)
    return jsonify({'doc_id': doc_id, 'plugin': plugin, 'items': items})

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

# Attribute-wise listing across documents
@app.get('/api/attr/<plugin>/docs')
def api_attr_docs(plugin):
    pdir = ATTR_DIR / plugin
    present = []
    if pdir.exists() and pdir.is_dir():
        for f in sorted(pdir.glob('*.json')):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                doc_id = data.get('doc_id') or f.stem
                val = data.get('value')
                preview = None
                if isinstance(val, str):
                    preview = val.strip().replace('\n',' ')[:160]
                present.append({'doc_id': doc_id, 'path': str(f), 'preview': preview})
            except Exception:
                doc_id = f.stem
                present.append({'doc_id': doc_id, 'path': str(f)})
    all_docs = _unique_doc_ids()
    have = {i['doc_id'] for i in present}
    missing = sorted([d for d in all_docs if d not in have])
    return jsonify({'plugin': plugin, 'present': present, 'missing': missing})

def _stop_worker_id(wid: int) -> bool:
    w = WORKERS.get(wid)
    if not w:
        return False
    proc = w.get('proc')
    if proc and (proc.poll() is None):
        try:
            proc.terminate()
        except Exception:
            pass
    WORKERS.pop(wid, None)
    return True

@app.get('/api/status')
def api_status():
    force_models = bool(request.args.get('force_models'))
    data = _read_jobs_status()
    workers = []
    for wid, w in WORKERS.items():
        proc = w.get('proc')
        workers.append({
            'id': wid,
            'pid': proc.pid if proc else None,
            'running': (proc is not None and proc.poll() is None),
            'plugins': w.get('plugins'),
            'batch': w.get('batch'),
            'watch': w.get('watch'),
            'started_at': w.get('started_at'),
            'log_tail': list(w.get('log', deque()))[-50:],
            'current': w.get('current'),
        })
    data['workers'] = workers
    if workers:
        data['worker'] = workers[0]
    try:
        cfg = load_configs()
        data['plugin_defaults'] = cfg.get('plugins', {})
    except Exception:
        data['plugin_defaults'] = {}
    data['llm'] = _llm_health(force=force_models)
    return jsonify(data)

@app.get('/')
def root():
    index_path = pathlib.Path(app.static_folder) / 'index.html'
    if not index_path.exists():
        return jsonify({'error':'index not found', 'path': str(index_path)}), 500
    return send_from_directory(app.static_folder, 'index.html')

@app.get('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051, debug=False, threaded=True)

