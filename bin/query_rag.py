import argparse
from kn.retrieval import answer_query
from kn.config import load_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--scope", default=None)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()
    cfg = load_configs()
    print(answer_query(args.q, cfg, scope=args.scope, topk=args.topk))