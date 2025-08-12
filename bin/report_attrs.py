import json, csv, pathlib

def main():
    base = pathlib.Path('.knowledge/indexes/attributes')
    out = pathlib.Path('.knowledge/exports/reports')
    out.mkdir(parents=True, exist_ok=True)

    def load(name: str):
        result = {}
        p = base / name
        if p.exists():
            for f in p.glob('*.json'):
                try:
                    rec = json.loads(f.read_text(encoding='utf-8'))
                    result[rec.get('doc_id')] = rec
                except Exception:
                    continue
        return result

    summaries = load('summary-20w')
    tags = load('topic-tags')
    pii = load('pii-scan')
    ids = sorted(set(summaries) | set(tags) | set(pii))

    rows = []
    for i in ids:
        rows.append({
            'doc_id': i,
            'summary': summaries.get(i, {}).get('value', ''),
            'tags': tags.get(i, {}).get('value', ''),
            'pii_risk': pii.get(i, {}).get('risk', ''),
        })

    csv_path = out / 'attributes_catalog.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['doc_id','summary','tags','pii_risk'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] wrote {csv_path}")

if __name__ == '__main__':
    main()


