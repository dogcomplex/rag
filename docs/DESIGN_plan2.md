# Comprehensive Delivery Plan

## 0. Scope & Intent

This plan merges the objectives from **DESIGN.md** and **DESIGN_additions.md** with the current repository state. It establishes a practical, phased roadmap that covers ingestion, chunking, retrieval, graph construction, attribute enrichment, dashboard operations, CorrectiveRAG-style refinements, hierarchical summarisation, and long-term automation. Every suggestion from the design corpus is evaluated, scoped, and placed within an actionable sequence.

## 1. Architecture Baseline (from DESIGN.md)

- **Environment**: Windows 10 Pro, LM Studio default, optional RTX 5090/3090 usage, offline-first.
- **Sidecar storage**: `.knowledge/` houses chunks, embeddings, graph, attributes, exports, queues.
- **Document identity**: SHA256 doc hashes, chunk IDs `docid-XXXX`.
- **Ingestion**: `initial_scan` + `chunk_repo`; current chunking is paragraph-based with overlap.
- **Embeddings**: SentenceTransformers + hnswlib index, nightly rebuild recommended.
- **Graph**: NetworkX with naive entity co-occurrence; Louvain communities stored as JSONL.
- **Attributes**: Standalone Python scripts writing to `.knowledge/indexes/attributes/<plugin>/`.
- **Queue**: SQLite with retry metadata; `enrich_worker.py` handles batching and concurrency.
- **Dashboard**: Planner, ingest controls, worker controls, queue management, coverage view, LLM gateway control, docs/attributes explorer, failed jobs viewer.
- **Retrieval/export**: Dense + BM25 hybrid, monofile export with optional budget & PII exclusion.
- **Operational defaults**: watch daemon, periodic ingest, manual exports, prompt-level cache.

## 2. Suggestion Inventory (DESIGN_additions.md & conversation)

| Topic | Suggestion | Source | Status |
| --- | --- | --- | --- |
| Hierarchical summarisation | Multi-level chunk→section→doc→repo reducers; multiple hierarchies (file tree, domain, priority) | Added roadmap (lines 1466–1504) | Planned – implement via phases C/D |
| CorrectiveRAG (CRAG) | Retrieval grading, knowledge strips, web-search fallback, query rewriting | Early section (LangGraph CRAG walkthrough) | Not implemented – requires new grader, strip refinement |
| Summary-first indexing | Build doc summaries (~15% length) for retrieval, secondary index, rehydration | “Retrieval-Augmented reasoning with lean LLMs” notes | Not implemented |
| Retrieval as a tool | Gate retrieval (should/should-not), auto query rewrite on weak hits | Same section | Not implemented |
| CRAG grading thresholds | Config knobs for upper/lower bounds, cross-encoder | Same section | Not implemented |
| Two-tier k selection | Pre-calc p@k, run small k by default, expand on fallback | Same section | Not implemented |
| Test-time scaling | Self-consistency voting, budget forcing | Same section | Not implemented |
| Conversation assembly | Inject context per turn instead of history bloat | Same section | Not implemented |
| Evaluation tooling | CLI to measure retrieval metrics (p@k etc.) | Same section | Not implemented |
| Summary-first ingestion | Doc-level skeleton/outline feeding other plugins | Already partly done (doc-skeleton) | Needs deeper integration |
| Chunking improvements | Heading-aware segmentation, type policies | DESIGN.md future upgrades | Not implemented |
| Graph upgrades | Structured entities, relation typing | DESIGN.md future upgrades | Not implemented |
| Queue backend upgrade | Optionally move to Redis later | DESIGN.md future upgrades | Deferred |
| Export enhancements | Per-chunk citations, node exports | DESIGN.md known limitations | Not implemented |
| Security/PII | Expand PII detection & filtering | DESIGN.md future upgrades | Partial (regex plugin) |

## 3. Guiding Principles

1. **Hierarchical-first**: all enrichment operates at chunk → section → doc → hierarchy levels to avoid context overflows.
2. **Quality control**: integrate retrieval grading and knowledge refinement before generation.
3. **Offline-friendly**: maintain LM Studio compatibility; external tools optional and gated.
4. **Observability**: every new stage emits metrics (coverage, latency, failures) into the dashboard.
5. **Modularity**: reducers, graders, and hierarchies are independent jobs with clear inputs/outputs.

## 4. Holistic Roadmap

### Phase 0 – Baseline Hardening (Weeks 0–1)
**Objective**: Ensure existing pipeline is stable and observable.
- Finalise force-run behaviour (done) and log metadata (latency, token counts) per job.
- Document current coverage (chunk summaries, doc reduces) as baseline metrics.
- Add instrumentation hooks for retrieval (track top-k, source docs) without changing behaviour.

### Phase 1 – Chunking & Summary-First Index (Weeks 2–3)
- **Heading-aware chunker**: parse Markdown headings / PDF TOC to create structured sections; store `structure/<doc>.json`.
- **Chunk metadata**: include section IDs, breadcrumbs, sequence, chunk hash.
- **Doc summary plugin**: `doc-summary` produces ~15% length summary; stored in `.knowledge/indexes/summaries/doc/`.
- **Summary index**: secondary HNSW built from doc summaries, toggle via config.
- **Rehydration logic**: retrieval pipeline fetches summary hits then rehydrates either full doc or top chunks.

### Phase 2 – Automated Map→Section→Doc Reduce (Weeks 4–6)
- **Automatic chunk-summary runs** triggered on ingest/update (dashboard toggle + CLI).
- **Section reducers**: aggregate chunk summaries, tags, glossary entries per section.
- **Doc reducers**: consume sections first, fallback to chunks; emit provenance.
- **Dependency tracker**: maintain DAG (chunk→section→doc) for targeted rebuilds.
- **Dashboard coverage**: show chunk/section/doc status, stale counts, mtime.

### Phase 3 – CorrectiveRAG & Retrieval Enhancements (Weeks 7–10)
- **Retrieval grader**: integrate local cross-encoder to label relevant/ambiguous/irrelevant chunks.
- **Strip refinement**: partition relevant chunks into strips (≈300 chars), grade, keep top strips.
- **Fallback logic**: if retrieval weak, rewrite query (local LLM) and/or expand dense k, optionally hop graph neighbours.
- **Should-retrieve gate**: logistic or cross-encoder to skip retrieval for trivial questions.
- **Two-tier k**: default small k; escalate on fallback. Provide CLI to benchmark p@k.
- **Self-consistency & budget forcing**: optional multi-pass generation with voting on answers.
- **Conversation assembly**: restructure chat prompts to inject fresh context as system message per turn.

### Phase 4 – Hierarchies & Repository Rollups (Weeks 11–14)
- **Hierarchy registry**: define base hierarchies (file tree, domain, curated priority) as JSON.
- **Node rollups**: summarise each hierarchy node using section/doc outputs.
- **Dashboard tree**: visualise hierarchy, show summary, freshness, run controls.
- **API endpoints**: `/api/hierarchy/<name>` for agent/export integration.

### Phase 5 – Retrieval & Export Deepening (Weeks 15–18)
- **Hierarchy-aware retrieval**: incorporate node summaries + weights during scoring; optional reranker.
- **Graph upgrades**: replace naive entity extraction with structured pipeline (NER + patterns); map entities to hierarchy nodes; update community summaries.
- **Export**: generate node-level reports with citations to chunk IDs, include PII filtering and rehydrated content.

### Phase 6 – Automation, Evaluation, Security (Weeks 19–22)
- **Scheduler**: scripts & dashboard jobs for nightly chunk refresh, weekly hierarchy rebuild, monthly evaluation.
- **Evaluation CLI**: `bin/eval_retrieval.py` to compute p@k, accuracy; feed results to dashboard.
- **Telemetry**: capture token usage, latency, failure counts; chart in dashboard.
- **Enhanced PII**: optional LLM-assisted PII detection; integrate into exports & retrieval gating.
- **Security posture**: document offline boundaries, optional encryption at rest for `.knowledge/users/`.

### Phase 7 – Future Extensions (post roadmap)
- **Redis/FAISS/Qdrant** swap readiness.
- **vLLM/Ollama** integration guide once Windows constraints ease.
- **Agent planner** that navigates hierarchies and triggers focused reruns.
- **User profile personalisation** leveraging future `profile.yml` + event imports (from DESIGN_additions).

## 5. Cross-Cutting Concerns

### Testing & QA
- Golden doc corpus (small/medium/large) for functional tests (chunking, reducers, retrieval, exports).
- Unit tests per plugin with mock inputs (ensuring chunk-size budgets).
- Benchmark suite for retrieval gating, CRAG accuracy, self-consistency effects.

### Observability
- Extend dashboard with metrics tab: job runtimes, queue depth, p@k trend, LM Studio latency.
- Structured logs emitted to `.knowledge/logs/pipeline_metrics.jsonl` for offline analysis.

### Operations
- CLI wrappers for manual rebuilds (`bin/hierarchy_build.py`, `bin/hierarchy_status.py`).
- Document runbooks (watcher, ingestion, worker restart, LM Studio issues).
- Clear cache management (prompt cache, chunk summaries) with TTLs.

## 6. Risk Assessment

| Risk | Impact | Mitigation |
| --- | --- | --- |
| LM Studio limitations | Throughput bottlenecks | Enforce chunk-first workflow, throttle concurrency, consider remote fallback flag |
| Increased job volume | Longer rebuild times | Dependency graph for selective reruns, schedule heavy tasks |
| Complexity of reducers | Debug difficulty | Layered testing, provenance recording, CLI inspection tools |
| Retrieval grading errors | Filtering useful docs | Tune thresholds, allow bypass, log grading outcomes |
| Hierarchy drift | Stale summaries | Freshness metrics, automated rebuild cadence |

## 7. Success Metrics

- Coverage percentages across chunk/section/doc/hierarchy levels.
- Retrieval precision/recall (p@k) before/after CRAG & summary index.
- Average tokens per doc-level attribute vs. baseline (should drop).
- Job latency per stage, failure rates, queue wait times.
- Dashboard usage: map-reduce runs, hierarchy views, manual reruns.
- Qualitative user satisfaction: targeted feedback after each phase.

## 8. Immediate Next Actions

1. Socialise roadmap with stakeholders; agree on resourcing for Phases 1–3.
2. Stand up baseline metrics (coverage, runtimes) to compare post-Phase 1.
3. Create tickets for Phase 1 tasks (heading-aware chunker, doc-summary plugin, summary index, rehydration).
4. Define test corpus and evaluation methodology (labelled questions, ground-truth docs).

---

This plan integrates every major recommendation from the design corpus—hierarchical processing, CRAG-style retrieval hygiene, summary-first indexing, reranking, conversation management, evaluation tooling, and forward-looking extensions—while respecting the Windows + LM Studio constraints. Phased delivery keeps the system stable, observable, and continuously valuable.

