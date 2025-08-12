import argparse, pathlib
from kn.config import load_configs
from kn.file_index import initial_scan
from kn.chunking import chunk_repo
from kn.embeddings import embed_chunks
from kn.graph_nx import build_or_update_graph, summarize_communities

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_configs()
    repo = pathlib.Path(args.repo).resolve()
    print("[ingest] scanning repo…")
    docs = initial_scan(repo, cfg)
    print(f"[ingest] {len(docs)} docs found")
    print("[chunk] building chunks…")
    chunks = chunk_repo(docs, cfg, repo_root=repo)
    print("[embed] embedding chunks…")
    embed_chunks(chunks, cfg)
    print("[graph] updating graph…")
    build_or_update_graph(chunks, cfg)
    print("[summarize] community summaries…")
    summarize_communities(cfg)
    print("[done]")