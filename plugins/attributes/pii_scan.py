import sys, json, re, pathlib

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/pii-scan')
OUTDIR.mkdir(parents=True, exist_ok=True)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
CREDIT = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def score_flags(text):
    findings = []
    for m in EMAIL.findall(text or ""):
        findings.append({"type": "email", "value": m, "severity": 0.5})
    for m in PHONE.findall(text or ""):
        findings.append({"type": "phone", "value": m, "severity": 0.4})
    for m in CREDIT.findall(text or ""):
        findings.append({"type": "credit", "value": m, "severity": 0.9})
    risk = max((f["severity"] for f in findings), default=0.0)
    return findings, risk


for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text', '') or ""
    findings, risk = score_flags(text)
    OUTDIR.joinpath(f"{job['doc_id']}.json").write_text(
        json.dumps(
            {
                "doc_id": job['doc_id'],
                "attribute": "pii-scan",
                "value": findings,
                "risk": risk,
                "confidence": 0.6,
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    print(json.dumps({"status": "ok", "doc_id": job['doc_id']}))

