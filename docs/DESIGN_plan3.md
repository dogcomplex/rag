# GraphRAG Next Phases – Plan v3

This plan translates the current codebase and design docs into a prioritized, phased roadmap. It emphasizes Windows-first local operation, modularity, and measurable acceptance criteria. It assumes the present stack: LM Studio gateway + SQLite job queues + attribute plugins + NetworkX graph + HNSW (or CPU fallback) + Flask dashboard.

## 0) Current capabilities (code audit)

- Dashboard (`bin/dashboard_server.py` + `bin/dashboard_static/index.html`)
  - Job planning: `/api/plan` spawns `bin/plan_enqueue.py` with options (only-missing, limit, changed_since_min, map-reduce, doc_ids, summaries modes).
  - Direct enqueue: `/api/enqueue` writes into `.knowledge/queues/jobs.sqlite` via `kn.jobs_sqlite.enqueue`.
  - Worker control: `/api/worker/start|stop`, spawns `bin/enrich_worker.py` subprocesses, tails logs, shows current batch.
  - Queue visibility: `/api/queue/list`, `/api/queue/job/<id>`, `/api/queue/clear` (with reset-running/reset-counter modes).
  - LLM Gateway control: `/api/gateway/start|stop`, status tail, queue stats from `kn.llm_gateway.storage.QueueStorage`.
  - LLM health: via `kn.llm_gateway.client.submit_generic_request(action=health)`.
  - Docs/attributes: list docs, attribute coverage, view per-doc, per-attr, and chunk-level attribute outputs.

- Worker + jobs (MVP but solid foundation)
  - `kn.jobs_sqlite`: SQLite schema with jobs, retries, last_error, completed_at. Concurrency primitives via `limits`/`counters` and `try_acquire/release`.
  - `bin/enrich_worker.py`: batch dequeue by plugin, per-plugin `process_timeout`, global `max-inflight` concurrency gate, requeue on timeout/error, skip when output present unless force/overwrite, chunk-map and doc-reduce aware.
  - Planner `bin/plan_enqueue.py`: enqueues for multiple plugins, respects only-missing, supports summaries modes, changed_since cutoff, doc filter, map-reduce (chunk-summary then doc-reduce).

- LLM Gateway (separate queue)
  - `kn/llm_gateway`: SQLite-backed request/response queue, service agent registry (LM Studio), health endpoint, prompt-level cache hooks.

- Core pipeline
  - Ingest (`bin/ingest_build_graph.py`): scan → chunk → embed → graph → community summaries.
  - Retrieval/export (`kn/retrieval.py`, `bin/export_monofile.py`): dense-only baseline with community preface; BM25 planned/mentioned in docs.
  - Attribute plugins present: summaries (short/medium/long/outline), topic-tags, summary-20w, pii-scan, glossary, requirements, todo-items, faq-pairs, keyphrases, bridge-candidates, risk-scan, recent-summary, plus macro `multi-basic` and `doc-skeleton`.

## 1) Gaps and quick wins (surgical code fixes)

1. Dashboard → worker: UI exposes `max-inflight` but `/api/worker/start` doesn’t pass it to `enrich_worker.py`.
2. Queue list: `/api/queue/list` response lacks `last_error`; UI’s Failed Jobs section expects it.
3. Observability: `/api/status` summary in server lacks durations/throughput/ETA present in `bin/jobs_status.py` (front-end is already wired to show them).
4. Plugin defaults: front-end controls are disabled; no API to read/write `.knowledge/config/models.yml` `plugins:` block.
5. Watcher integration: `kn/file_index._Evt` leaves a TODO to enqueue downstream jobs on change.
6. LLM health: dashboard health relies on gateway call; ensure graceful error and model list surfacing even when gateway down.
7. Minor UI: coverage table columns misaligned (avg seconds cell displays wrong field), and small ergonomics (persist plugin selections done, doc filter OK).
8. Retrieval: no CRAG grading/strip refinement wired; dense-only baseline without BM25 merge in code.
9. Summary-first index and rehydration not present.
10. Orchestrator/trust/personalization not yet implemented (documented in DESIGN_additions.md).

Acceptance for 1–7: fixed endpoints with tests, dashboard shows durations/ETA, can adjust concurrency from UI, change events enqueue jobs.

---

## Phase A – Hardening and Observability (1–2 weeks)

Scope: Close gaps 1–7 to stabilize current UX and throughput.

- A1. Pass `--max-inflight` to worker
  - Edit `/api/worker/start` to forward `max_inflight` from JSON body to `_spawn_worker` and CLI arg.
  - Acceptance: Starting a worker with `max-inflight=3` shows counter limit set in logs; concurrent plugin runs obey global cap across multiple workers.

- A2. Queue diagnostics parity
  - Server: lift `durations`, `throughput`, `eta` logic from `bin/jobs_status.py` into dashboard server’s `_db_summary`.
  - Add `last_error` to `/api/queue/list` and keep `/api/queue/job/<id>` for detail.
  - Acceptance: Coverage panel shows avg seconds/DPS/ETA per plugin; Failed Jobs list shows `last_error` previews.

- A3. Plugin defaults API
  - Add `/api/plugins/config GET|POST` to read/update `plugins:` block in `.knowledge/config/models.yml` via `kn.config` merge/save helpers (write-back safe).
  - Enable front-end Load/Save to view and persist per-plugin `llm.model`, `timeout`, etc.
  - Acceptance: Change `summary-short` default model in UI, persist to YAML, reflected in worker `process_timeout` and LLM overrides.

- A4. Watcher → jobs bridge
  - In `kn/file_index._Evt.on_any_event`, enqueue lightweight jobs (e.g., recent-summary, topic-tags) and set a flag to schedule chunk/embed/graph refresh via nightly ingest (or enqueue chunk-summary map jobs if cheap policy).
  - Acceptance: Editing a file creates pending jobs visible in queue within 2s; worker processes them.

- A5. UI polish and resilience
  - Fix coverage table column alignment; ensure dashboard handles gateway being down with clear error (already partial).
  - Acceptance: No console errors; columns reflect correct metrics; health panel shows errors without breaking other panels.

---

## Phase B – Retrieval robustness (CRAG baseline) (1–2 weeks)

Goal: Add self-grading and strip-level refinement to improve answer quality without changing storage.

- B1. Local cross-encoder grader
  - New `kn/crag.py` with: `grade_chunks`, `decide_action`, `refine_strips`, `fallback_retrieve` (expand-k, optional graph hops, optional rewrite via local LLM), on-disk cache of grades.
  - Use `sentence-transformers` cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) by default.

- B2. Retrieval hook
  - Update `kn/retrieval.answer_query` to: initial retrieve → CRAG grade → action (Correct/Incorrect/Ambiguous) → strip refinement → assemble with community preface.
  - Config additions in `.knowledge/config/pipeline.yml`:
    - `crag.enabled`, `crag.upper`, `crag.lower`, `crag.topk_grade`
    - `crag_refine.strip_chars`, `crag_refine.max_strips_per_chunk`
    - `crag_fallback.expand_dense_k`, `crag_fallback.graph_hops`, `crag_fallback.rewrite_with_llm`, `crag_fallback.use_web=false`

- Acceptance
  - On known queries, CRAG filters irrelevant chunks; strip refinement reduces context size ≥30% with equal or better answers.
  - Config toggles CRAG without restarting; exporter notes CRAG path (Correct/Ambiguous/Incorrect) in metadata.

---

## Phase C – Summary-first retrieval and rehydration (1–2 weeks)

Goal: Improve first-hit relevance and cut context by indexing doc summaries.

- C1. Summary index
  - New `kn/summary_index.py`: build/read HNSW over per-doc summaries stored under `.knowledge/indexes/attributes/summaries*` or a dedicated `doc-summary` plugin directory.
  - Retrieval path (if enabled): query summary index → pick top docs → rehydrate to best chunks or full doc.

- C2. Config and CLI
  - `.knowledge/config/pipeline.yml`:
    - `summarize_index.enabled`, `summarize_index.target_ratio`, `summarize_index.use_for_retrieval`, `summarize_index.rehydrate: best_chunks|full_doc`.
  - Optionally extend `plan_enqueue` to ensure chosen summary modes exist (enqueue if missing).

- Acceptance
  - With summary-first enabled, top-k doc hit-rate improves on a labeled subset; context length reduced ≥40% at same answer quality.

---

## Phase D – Hierarchical summarization (map→reduce→rollups) (2–3 weeks)

Goal: Preserve nuance for long docs and enable corpus overviews.

- D1. Structure capture
  - Extend `kn/chunking.py` to persist document structure (section ids, levels, titles) to `.knowledge/indexes/structure/<doc_id>.json`.

- D2. Reducers
  - Add `plugins/attributes/chunk_summary.py` (chunk-level) and `plugins/attributes/doc_reduce.py` (merge over chunks/sections), with provenance.
  - Add corpus-level reducers for domains/repos; wire dashboard buttons to run reducers.

- D3. Export integration
  - `bin/export_monofile.py` adds pack modes: hierarchy-first, topic-first, claims-first (later), with quotes/citations.

- Acceptance
  - For a long doc, section-level outputs exist with incremental rebuild when a subset of chunks change; corpus overview pages render under `exports/reports/`.

---

## Phase E – Orchestrator, trust, personalization (2–3 weeks)

Goal: Route queries to the right methods and support personal vs. objective modes.

- E1. Orchestrator
  - New `kn/orchestrator.py`: intent detection (simple rules), quality signals (CRAG), source mix → select pipeline steps (RAG, CRAG refine, HSum, timeline, verify).
  - Config rules in pipeline YAML (as per DESIGN_additions.md).

- E2. Trust/utility scoring
  - New `kn/trust.py`, `kn/prioritizer.py`: compute TrustScore and UtilityScore per doc; order enrichment and retrieval budget allocation.

- E3. Personalization
  - New `kn/profile.py`, `kn/personalizer.py`, per-user overlays and interest vectors under `.knowledge/users/<user_id>/`.
  - Retrieval re-ranks by PersonalRelevance unless mode=objective.

- Acceptance
  - CLI supports `--user` and `--mode personal|balanced|objective`; outputs include brief “why selected” ledger.

---

## Phase F – Source adapters (rolling)

Goal: Normalize heterogeneous dumps to canonical docs; keep offline.

- Minimal initial set: Reddit saved JSON, GitHub repo zips, arXiv PDFs, Twitter export, Discord/Signal exports.
- New `sources/*.py` modules + `bin/ingest_<source>.py` scripts to transform into canonical doc schema then write through existing chunker.

Acceptance: Drop a dump file/folder, run ingest script, see docs, attributes, and retrieval working end-to-end.

---

## Phase G – Evaluation and QA

- `bin/eval_retrieval.py`: measure p@k and context sizes against a labeled subset.
- Golden-answer checks for key questions; report regressions per commit.

Acceptance: Baseline metrics dashboard; CRAG/summary-first deltas visible.

---

## Phase H – Performance and reliability

- Concurrency: configurable per-plugin batching; smarter backoff on gateway timeouts; purge stale rows in gateway DB.
- Caching: prompt-level cache in gateway; result caches for CRAG grades.
- Windows ops: ensure PowerShell helpers (`setup.ps1`) cover new configs; document LM Studio quirks.

Acceptance: Sustained throughput without thrashing; dashboard throughput and ETA stable.

---

## Phase I – Safety and PII

- Upgrade `pii-scan` with optional LLM assist (local) and region patterns; add license scan for code.
- Export guard: flag/omit low-trust content for objective reports.

Acceptance: PII false-positive/negative rates sampled; exports respect `export.exclude_pii`.

---

## Milestones & deliverables

- M0 (end of Phase A): Dashboard parity + watcher bridge; plugin defaults editable; stable workers; metrics in UI.
- M1 (end of Phase B): CRAG on by default with config; quality wins on test set.
- M2 (end of Phase C): Summary-first index; rehydration; shorter contexts.
- M3 (end of Phase D): Hierarchical reducers; corpus reports.
- M4 (end of Phase E): Orchestrator + modes; basic personalization.
- M5 (rolling): 2–4 source adapters; ingest guides.

---

## Concrete edits (initial backlog)

These are small, high-impact edits to start immediately:

1) Dashboard → worker concurrency
   - dashboard_server: include `max_inflight` in `/api/worker/start` body, pass to `_spawn_worker` → `enrich_worker.py --max-inflight N`.

2) Queue diagnostics
   - dashboard_server: enhance `_db_summary` with durations/throughput/ETA; add `last_error` to `/api/queue/list` rows.

3) Plugin defaults API
   - dashboard_server: add `/api/plugins/config GET|POST`; implement YAML read/merge/write; enable front-end buttons.

4) Watcher bridge
   - kn/file_index: enqueue attribute jobs on change (cheap set) and optionally mark for nightly ingest.

5) CRAG module + retrieval hook
   - kn/crag.py; wire to `kn/retrieval.answer_query`; add config blocks.

6) Summary index
   - kn/summary_index.py; gate via config; integrate into retrieval pre-pass.

Definition of done for this plan: M0 and M1 shipped, with M2 queued; dashboard visibly shows metrics and editable plugin defaults; retrieval paths demonstrate measurable gains with CRAG and summary-first.


