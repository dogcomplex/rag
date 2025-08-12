import json, pathlib

def main():
    src = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')
    out_dir = pathlib.Path('.knowledge/exports/reports')
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding='utf-8').splitlines() if src.exists() else []
    md = ['# Communities', '']
    for l in lines[:200]:
        try:
            c = json.loads(l)
            md.append(f"- {c['id']} (size={c['size']}): {c['summary']}")
        except Exception:
            continue
    out = out_dir / 'communities.md'
    out.write_text('\n'.join(md), encoding='utf-8')
    print(f"[report] wrote {out}")

if __name__ == '__main__':
    main()

