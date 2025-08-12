import pathlib, json, networkx as nx
from collections import defaultdict
GRAPH_PATH = pathlib.Path('.knowledge/indexes/graph/graph.jsonl')
COMM_PATH = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')
G = nx.Graph()
def naive_entities(text: str):
    import re
    ents = set(re.findall(r"[A-Z][a-zA-Z0-9_]+|\b[A-Za-z0-9_]+\.\w+|#\w+", text))
    return list(ents)[:50]
def build_or_update_graph(chunks, cfg):
    for c in chunks:
        ents = naive_entities(c["text"])
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, kind='entity')
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                u, v = ents[i], ents[j]
                w = G[u][v]["weight"]+1 if G.has_edge(u,v) else 1
                G.add_edge(u, v, weight=w)
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open('w', encoding='utf-8') as f:
        for u, v, d in G.edges(data=True):
            f.write(json.dumps({"src": u, "dst": v, "weight": d.get("weight",1)})+"\n")
def summarize_communities(cfg):
    try:
        import community as community_louvain
        part = community_louvain.best_partition(G) if len(G) else {}
        comm_to_nodes = defaultdict(list)
        for n, c in part.items():
            comm_to_nodes[c].append(n)
        comms = [{"id": f"C{k}", "size": len(v), "summary": ", ".join(v[:10])} for k, v in comm_to_nodes.items()]
    except Exception:
        comms = []
        for idx, comp in enumerate(nx.connected_components(G)):
            nodes = list(comp)
            summary = ", ".join(nodes[:10])
            comms.append({"id": f"C{idx}", "size": len(nodes), "summary": summary})
    COMM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMM_PATH.open('w', encoding='utf-8') as f:
        for c in comms:
            f.write(json.dumps(c)+"\n")