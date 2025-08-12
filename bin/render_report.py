import json, pathlib, html, datetime as dt

CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')
ATTR_DIR   = pathlib.Path('.knowledge/indexes/attributes')
COMM_PATH  = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')
OUT_HTML   = pathlib.Path('.knowledge/exports/reports/index.html')
GRAPH_PATH = pathlib.Path('.knowledge/indexes/graph/graph.jsonl')

PLUGINS = ['summary-20w','topic-tags','pii-scan','glossary','requirements','todo-items','faq-pairs']

def _iter_docs():
    seen = {}
    for p in CHUNKS_DIR.glob('*.json'):
        try:
            rec = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        doc_id = rec.get('doc_id')
        if doc_id in seen:
            continue
        meta = rec.get('meta') or {}
        seen[doc_id] = {
            'doc_id': doc_id,
            'rel': meta.get('rel') or meta.get('path') or '',
            'domain': meta.get('domain') or 'root',
        }
    return list(seen.values())

def _load_attr(plugin, doc_id):
    p = ATTR_DIR / plugin / f'{doc_id}.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def _load_comms(max_items=100):
    out = []
    if COMM_PATH.exists():
        for i, line in enumerate(COMM_PATH.read_text(encoding='utf-8').splitlines()):
            if i >= max_items: break
            try:
                c = json.loads(line)
                out.append(c)
            except Exception:
                continue
    return out

def _esc(s):
    return html.escape(str(s or ''))

def _render_doc_card(d):
    doc_id = d['doc_id']
    parts = [f"<div class='card'><div class='hdr'><span class='rel'>{_esc(d['rel'])}</span><span class='meta'>doc:{doc_id} · domain:{_esc(d['domain'])}</span></div>"]
    # attributes
    def sec(title, txt):
        if not txt: return
        parts.append(f"<div class='sec'><div class='stit'>{_esc(title)}</div><div class='sbody'><pre>{_esc(txt)}</pre></div></div>")

    # summary
    s = _load_attr('summary-20w', doc_id); sec('Summary (20w)', s and s.get('value'))
    t = _load_attr('topic-tags', doc_id);  sec('Tags', t and t.get('value'))
    g = _load_attr('glossary', doc_id);    sec('Glossary', g and g.get('value'))
    r = _load_attr('requirements', doc_id);sec('Requirements', r and r.get('value'))
    f = _load_attr('faq-pairs', doc_id);   sec('FAQ', f and f.get('value'))
    td = _load_attr('todo-items', doc_id); sec('TODOs', json.dumps(td.get('value'), ensure_ascii=False, indent=2) if td else '')
    pii = _load_attr('pii-scan', doc_id)
    if pii:
        sec('PII risk', f"risk={pii.get('risk')} findings={len(pii.get('value') or [])}")
    parts.append("</div>")
    return "\n".join(parts)

def main():
    docs = sorted(_iter_docs(), key=lambda x: x['rel'].lower())
    comms = _load_comms()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    # prepare graph data (nodes/links), limited for performance
    graph_nodes = []
    graph_links = []
    if GRAPH_PATH.exists():
        # Build nodes/links
        name_to_idx = {}
        degree = {}
        edges = []
        for i, line in enumerate(GRAPH_PATH.read_text(encoding='utf-8').splitlines()):
            try:
                e = json.loads(line)
            except Exception:
                continue
            edges.append((e.get('src'), e.get('dst'), int(e.get('weight', 1))))
            if len(edges) >= 1500:  # cap edges to keep UI responsive
                break
        for u, v, w in edges:
            for n in (u, v):
                if n not in name_to_idx:
                    idx = len(graph_nodes)
                    name_to_idx[n] = idx
                    graph_nodes.append({'id': n, 'deg': 0, 'idx': idx})
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            graph_links.append({'source': name_to_idx[u], 'target': name_to_idx[v], 'weight': w})
        for n in graph_nodes:
            n['deg'] = degree.get(n['id'], 0)
    # simple HTML
    graph_json = json.dumps({'nodes': graph_nodes, 'links': graph_links}, ensure_ascii=False)
    head = f"""
<!doctype html>
<html><head><meta charset='utf-8'>
<title>Knowledge Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#fafafa;color:#222}}
header{{background:#0f62fe;color:#fff;padding:12px 16px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:18px;margin:0}}
.sub{{opacity:.9;font-size:12px}}
.wrap{{padding:14px 16px}}
.row{{display:flex;gap:16px;flex-wrap:wrap}}
.col{{flex:1 1 380px;min-width:320px}}
.card{{background:#fff;border:1px solid #eee;border-radius:8px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.hdr{{display:flex;justify-content:space-between;gap:12px;padding:10px 12px;border-bottom:1px solid #f0f0f0}}
.rel{{font-weight:600}}
.meta{{font-size:12px;color:#666}}
.sec{{padding:10px 12px;border-top:1px dashed #f2f2f2}}
.stit{{font-size:12px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:.02em}}
.sbody pre{{white-space:pre-wrap;margin:0;font-family:ui-monospace,Consolas,monospace;font-size:13px;line-height:1.35}}
.pill{{display:inline-block;background:#eef5ff;color:#0f62fe;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}}
#q{{width:100%;padding:10px;border:1px solid #ddd;border-radius:8px;margin:10px 0}}
</style>
</head><body>
<script>window.GRAPH_DATA = {graph_json};</script>
<header><h1>Knowledge Report</h1><div class='sub'>generated {now}</div></header>
<div class='wrap'>
<input id='q' placeholder='Filter by file, domain or text... (client-side)'>
<div class='row'>
  <div class='col'>
    <div class='card'>
      <div class='hdr'><span class='rel'>Communities</span><span class='meta'>{len(comms)} items</span></div>
      <div class='sec'>
        {''.join(f"<div class='pill'>"+_esc(c.get('id'))+f" size={_esc(c.get('size'))}</div>" for c in comms)}
      </div>
      <div class='sec'>
        <div class='stit'>Top community keywords</div>
        <div class='sbody'>
          <ul>
          {''.join(f"<li>- { _esc(c.get('id')) } (size={ _esc(c.get('size')) }): { _esc(c.get('summary')) }</li>" for c in comms)}
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div class='col'>
    <div class='card'>
      <div class='hdr'><span class='rel'>Graph (preview)</span><span class='meta'>{len(graph_nodes)} nodes · {len(graph_links)} edges</span></div>
      <div class='sec'>
        <div id='graphwrap' style='width:100%;height:380px;'>
          <svg id='graph' width='100%' height='360'></svg>
        </div>
        <div style='font-size:12px;color:#666'>Drag to move. Hover to highlight neighborhood. Filter: min degree <input type='range' id='mindeg' min='0' max='10' value='0'/> <span id='mindegv'>0</span></div>
      </div>
    </div>
  </div>
</div>
<div id='docs'>
"""
    cards = [ _render_doc_card(d) for d in docs ]
    tail = """
</div></div>
<script>
// Try local d3 first, then CDN; if all fail, render static graph
(function(){
  function onReady(){ if (typeof d3!=='undefined') { initGraph(); } else { renderStaticGraph(); } }
  function tryCdn(){
    var s=document.createElement('script'); s.src='https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js';
    s.onload=onReady;
    s.onerror=function(){
      var s2=document.createElement('script'); s2.src='https://unpkg.com/d3@7/dist/d3.min.js'; s2.onload=onReady; s2.onerror=onReady; document.head.appendChild(s2);
    };
    document.head.appendChild(s);
  }
  var sl=document.createElement('script'); sl.src='d3.min.js'; sl.onload=onReady; sl.onerror=tryCdn; document.head.appendChild(sl);
  window.addEventListener('load', onReady);
})();

// guard flags to prevent double rendering
window.__graphInited = window.__graphInited || false;
window.__staticRendered = window.__staticRendered || false;

const GRAPH_DATA = (window.GRAPH_DATA && window.GRAPH_DATA.nodes) ? window.GRAPH_DATA : {nodes:[], links:[]};

function initGraph(){
  if (window.__graphInited) return; window.__graphInited = true;
  const svg = d3.select('#graph');
  if (!svg.node()) return;
  if (GRAPH_DATA.nodes.length===0){
    const w=document.getElementById('graphwrap');
    if(w){ w.innerHTML = "<div style='padding:8px;color:#666;font-size:12px'>No graph data available.</div>"; }
    return;
  }
  // clear any prior content (e.g., static fallback)
  svg.selectAll('*').remove();
  const wrap = document.getElementById('graphwrap');
  const width = (wrap && wrap.clientWidth) ? wrap.clientWidth : 640;
  const height = 360;
  svg.attr('viewBox', `0 0 ${width} ${height}`).attr('width', width).attr('height', height);
  const g = svg.append('g');
  const zoom = d3.zoom().on('zoom', (ev)=> g.attr('transform', ev.transform));
  svg.call(zoom);

  const color = d3.scaleOrdinal(d3.schemeCategory10);
  const deg = GRAPH_DATA.nodes.map(n=>n.deg);
  const maxDeg = Math.max(1, d3.max(deg));
  const rscale = d3.scaleSqrt().domain([0, maxDeg]).range([3, 14]);

  const nodes = GRAPH_DATA.nodes.map(d=>Object.assign({}, d));
  const rawLinks = GRAPH_DATA.links.map(d=>Object.assign({}, d));
  // Build link objects referencing node objects by index to avoid id mismatches
  const linkData = rawLinks.map(l=>({source: nodes[l.source], target: nodes[l.target], weight: l.weight}));

  // Seed positions in a circle for visibility before forces settle
  const R = Math.min(width, height) * 0.45;
  nodes.forEach((d,i)=>{ const a = (2*Math.PI*i)/nodes.length; d.x = width/2 + R*Math.cos(a); d.y = height/2 + R*Math.sin(a); });

  const link = g.append('g').attr('stroke', '#999').attr('stroke-opacity', 0.35)
      .selectAll('line').data(linkData).join('line').attr('stroke-width', d=>Math.max(1, Math.log(1+(d.weight||1))));
  const node = g.append('g').attr('stroke', '#fff').attr('stroke-width', 1)
      .selectAll('circle').data(nodes).join('circle')
      .attr('r', d=>rscale(d.deg)).attr('fill', (d,i)=> color(i%10)).call(drag(sim()));
  const label = g.append('g').selectAll('text').data(nodes).join('text')
      .text(d=>d.id).attr('font-size', '10px').attr('fill', '#444').attr('pointer-events', 'none');

  function sim(){
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(linkData).distance(40).strength(0.2))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(width/2, height/2))
      .alpha(1).on('tick', ticked);
    function ticked(){
      link.attr('x1', d=>d.source.x).attr('y1', d=>d.source.y).attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
      node.attr('cx', d=>d.x).attr('cy', d=>d.y);
      label.attr('x', d=>d.x+6).attr('y', d=>d.y+3);
    }
    return simulation;
  }

  function drag(simulation){
    function dragstarted(event, d){ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }
    function dragged(event, d){ d.fx=event.x; d.fy=event.y; }
    function dragended(event, d){ if (!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }
    return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
  }

  node.on('mouseover', (_, d)=>{
    const neigh = new Set([d.index]);
    linkData.forEach(l=>{ if(l.source.index===d.index) neigh.add(l.target.index); if(l.target.index===d.index) neigh.add(l.source.index); });
    node.attr('opacity', n=> neigh.has(n.index)?1:0.15);
    link.attr('opacity', l=> l.source.index===d.index||l.target.index===d.index?0.8:0.1);
    label.attr('opacity', n=> neigh.has(n.index)?1:0.1);
  }).on('mouseout', ()=>{
    node.attr('opacity', 1); link.attr('opacity', 0.35); label.attr('opacity', 1);
  });

  const slider = document.getElementById('mindeg');
  const sv = document.getElementById('mindegv');
  if (slider){
    slider.max = String(Math.max(10, Math.floor(maxDeg)));
    slider.addEventListener('input', ()=>{
      sv.textContent = slider.value;
      const minDeg = +slider.value;
      node.attr('display', d=> d.deg>=minDeg? null : 'none');
      label.attr('display', d=> d.deg>=minDeg? null : 'none');
      link.attr('display', l=> (l.source.deg>=minDeg && l.target.deg>=minDeg)? null : 'none');
    });
  }
  // Fallback: if nothing rendered for any reason, draw static
  // do not auto-render static here; static is reserved for no-d3 case
}

function renderStaticGraph(){
  if (window.__graphInited || window.__staticRendered) return; window.__staticRendered = true;
  const svg = document.getElementById('graph');
  const wrap = document.getElementById('graphwrap');
  if (!svg || !wrap) return;
  const width = (wrap && wrap.clientWidth) ? wrap.clientWidth : 640;
  const height = 360;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const ns = 'http://www.w3.org/2000/svg';
  // Clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const nodes = GRAPH_DATA.nodes.slice(0, 500); // cap
  const links = GRAPH_DATA.links.slice(0, 1500);
  const R = Math.min(width, height) * 0.42;
  nodes.forEach((d,i)=>{ const a = (2*Math.PI*i)/nodes.length; d.x = width/2 + R*Math.cos(a); d.y = height/2 + R*Math.sin(a); });
  // draw links
  links.forEach(l=>{
    const s = nodes[l.source]; const t = nodes[l.target]; if(!s||!t) return;
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
    line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
    line.setAttribute('stroke', '#bbb'); line.setAttribute('stroke-opacity', '0.5'); line.setAttribute('stroke-width', String(Math.max(1, Math.log(1+(l.weight||1)))));
    svg.appendChild(line);
  });
  // draw nodes + labels
  nodes.forEach((d,i)=>{
    const circ = document.createElementNS(ns, 'circle'); circ.setAttribute('cx', d.x); circ.setAttribute('cy', d.y); circ.setAttribute('r', String(3 + Math.min(12, (d.deg||0)/2))); circ.setAttribute('fill', '#0f62fe'); circ.setAttribute('stroke', '#fff'); circ.setAttribute('stroke-width','1'); svg.appendChild(circ);
    if (i < 400){ const tx = document.createElementNS(ns, 'text'); tx.setAttribute('x', d.x+6); tx.setAttribute('y', d.y+3); tx.setAttribute('font-size','10'); tx.setAttribute('fill','#444'); tx.textContent = d.id; svg.appendChild(tx); }
  });
}

const q = document.getElementById('q');
q.addEventListener('input', () => {
  const term = q.value.toLowerCase();
  document.querySelectorAll('#docs .card').forEach(card => {
    card.style.display = card.innerText.toLowerCase().includes(term) ? '' : 'none';
  });
});
</script>
</body></html>
"""
    html_out = head.replace('__NODES__', json.dumps(graph_nodes, ensure_ascii=False)) \
                  .replace('__LINKS__', json.dumps(graph_links, ensure_ascii=False)) \
               + "\n".join(cards) + tail
    OUT_HTML.write_text(html_out, encoding='utf-8')
    print(f"[report] wrote {OUT_HTML}")

if __name__ == '__main__':
    main()

