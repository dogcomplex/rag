param(
  [string]$RepoPath = ".\FAE",
  [string]$Model = "llama-3.1-8b-instruct"
)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
} else {
  Write-Host "Virtual environment already exists; skipping creation."
}

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
  throw "Virtual environment missing activation script; rerun without workers or recreate manually."
}

. .\.venv\Scripts\Activate.ps1

if (Test-Path "requirements-win.txt") {
  python -m pip install -r requirements-win.txt
} else {
  python -m pip install -r requirements.txt
}
$env:PYTHONPATH = $PSScriptRoot

try {
  $resp = Invoke-RestMethod -Method Get -Uri "http://localhost:1234/v1/models" -TimeoutSec 5
} catch { Write-Warning "LM Studio local server not reachable." }

@"
OPENAI_BASE_URL=http://10.5.0.2:12345/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=$Model
EMBED_MODEL=BAAI/bge-small-en-v1.5
REPO_PATH=$RepoPath
KN_ROOT=.knowledge
OCR_ENABLED=false
"@ | Set-Content .env -Encoding UTF8

python .\bin\ingest_build_graph.py --repo $RepoPath --full