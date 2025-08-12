import argparse, pathlib
from kn.config import load_configs
from kn.file_index import watch_changes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to watch")
    args = ap.parse_args()
    cfg = load_configs()
    repo = pathlib.Path(args.repo).resolve()
    print(f"[watch] monitoring {repo} … (Ctrl+C to quit)")
    watch_changes(repo, cfg)