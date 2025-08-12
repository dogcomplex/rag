import argparse, pathlib
from kn.exporter import export_monofile
from kn.config import load_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--scope", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_configs()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_monofile(args.q, cfg, out, budget=args.budget, scope=args.scope)
    print(f"[export] wrote {out}")