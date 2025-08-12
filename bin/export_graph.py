import json, pathlib, networkx as nx

def main():
    graph_path = pathlib.Path('.knowledge/indexes/graph/graph.jsonl')
    out_dir = pathlib.Path('.knowledge/exports/graph')
    out_dir.mkdir(parents=True, exist_ok=True)
    G = nx.Graph()
    if graph_path.exists():
        for l in graph_path.read_text(encoding='utf-8').splitlines():
            try:
                d = json.loads(l)
                G.add_edge(d['src'], d['dst'], weight=d.get('weight', 1))
            except Exception:
                continue
    nx.write_gexf(G, str(out_dir / 'graph.gexf'))
    nx.write_graphml(G, str(out_dir / 'graph.graphml'))
    print(f"[graph] wrote {out_dir / 'graph.gexf'} and {out_dir / 'graph.graphml'}")

if __name__ == '__main__':
    main()

