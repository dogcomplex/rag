# GraphRAG Project Delivery Plan

## 0. Executive Summary

The repository now delivers a Windows-first GraphRAG pipeline with a live dashboard, LM Studio integration, enriched job queue, and robust attribute plugins. Foundational chunking and map–reduce scaffolding exist (`chunk-summary`, `doc-reduce`), but hierarchical processing, repository-wide rollups, and retrieval upgrades remain aspirational in the design docs. This plan consolidates DESIGN.md and DESIGN_additions.md into a realistic build path with staged milestones, risk controls, and success metrics.

## 1. Current Capability Assessment

| Area | Current State | Strengths | Gaps / Debt |
| --- | --- | --- | --- |
| **Ingestion & Chunking** | `kn/chunking.py` paragraph splitter with overlap; per-chunk JSON persists sequence metadata. | Stable sidecar storage, doc-level & chunk-level IDs. | No heading-aware segmentation, no section hierarchy files, limited chunk policies. |
| **Attributes** | Rich set of doc-level plugins (summaries, tags, glossary, requirements, multi-basic, etc.) + chunk/document reduce scripts. | Plugins modular, now support force overwrite; dashboard controls exist. | Most plugins operate on full doc text; chunk outputs underused; no reducers for non-summary attributes. |
| **Map-Reduce** | `chunk-summary` + `doc-reduce` optional via `plan_enqueue.py --map-reduce`. | Proven pattern for summaries. | Manual trigger only; no automation or dashboard toggle; lacks section-level aggregation. |
| **Dashboard** | Comprehensive monitoring, enqueue, queue control, LLM health, doc views, force-run. | Excellent observability and manual control. | No hierarchy visualisation, no chunk coverage view, limited alerts. |
| **Retrieval** | Dense HNSW + BM25 fallback; community preface. | Works for MVP queries. | No hierarchical retrieval, reranker, or chunk summary usage. |
| **Graph** | NetworkX with naive entity extraction, Louvain when available. | Lightweight and replaceable. | Needs structured entities, relation typing, integration with chunk hierarchies. |
| **Operations** | Robust worker loop, concurrency guard, LM Studio integration, response caching. | Stable foundation for throughput. | No pipeline orchestration for higher-level workflows (nightly hierarchy rebuild, etc.). |

## 2. Gap Analysis vs. Design Documents

| Design Expectation (DESIGN.md & additions) | Implementation Status | Delta |
| --- | --- | --- |
| Hierarchical chunking with per-section metadata | Partial – chunk IDs + seq only | Need heading detection, hierarchy JSON, multi-policy chunker |
| Automatic map→reduce summaries with dashboard control | Partially manual | Add watcher hooks / scheduled jobs and dashboard triggers |
| Multi-level summaries (chunk → section → doc → domain) | Not yet | Build reducers per level, maintain provenance |
| Attribute aggregation over chunks | Missing except doc summary | Extend plugins (tags, glossary, requirements, risk) with chunk map & reduce |
| Repository-level hierarchies (file tree, priority, recency) | Not present | Design hierarchy registry + rollup storage |
| Retrieval using hierarchy metadata | Absent | Update retriever/exporter to leverage section summaries |
| Graph enriched by chunk entities | Minimal | Integrate chunk outputs; move to structured extraction |
| Agent planner & export improvements | Not begun | Downstream once hierarchy data stabilizes |

## 3. Guiding Principles

1. **Incremental adoption** – maintain working pipeline while layering richer hierarchy features.
2. **Deterministic sidecar outputs** – all hierarchical artifacts live in `.knowledge/` with timestamps & provenance for reproducibility.
3. **Composable stages** – treat chunk, section, doc, hierarchy reducers as independent jobs to support re-runs and failure isolation.
4. **Observability first** – every stage emits metrics (counts, durations, coverage) to the dashboard before scaling out.
5. **Context guardrails** – no plugin should request text beyond configured token budgets; chunk-first strategy enforced by default.

## 4. Roadmap & Priorities

### Phase A – Foundation Hardening (Weeks 1–2)

1. **Chunking 2.0**
   - Add heading-/structure-aware splitter (Markdown/pseudo-outline) to derive `section_id`, `title`, `level`.
   - Persist `structure/<doc_id>.json` describing the hierarchy tree.
   - Update chunk metadata with `section_id`, `breadcrumbs`, and chunk-level hashes.
2. **Map-Reduce Automation**
   - Extend ingest/watch pipeline to enqueue `chunk-summary` automatically after chunk writes.
   - Integrate map-reduce toggle in dashboard planner (checkbox enabling chunk + doc reduce). 
   - Record outputs in new manifest (e.g., `.knowledge/indexes/manifests/attributes.jsonl`).
3. **Force-run Observability**
   - Dashboard: add per-attribute status columns (last run, last error). 
   - Worker: emit structured logs (JSON) capturing doc_id, plugin, runtime, token counts.

**Deliverables:** new chunk structure files, automated chunk summary pipeline, improved dashboard metrics.

### Phase B – Section & Doc Aggregation (Weeks 3–5)

1. **Section Reducers**
   - Implement `plugins/attributes/section_summary.py` to merge chunk summaries per section.
   - Build parallel reducers for glossary/tags/requirements (with merge heuristics: frequency, TF-IDF, dedupe).
2. **Doc Reducer Upgrade**
   - Update `doc_reduce` to consume section outputs first, falling back to chunk-level if missing.
   - Emit doc-level outline referencing section IDs for navigation.
3. **Incremental Update Engine**
   - When chunk file changes, re-run downstream jobs selectively (chunk → section → doc). 
   - Store dependency graph (chunk → section, section → doc) for targeted rebuild.
4. **Coverage Dashboard**
   - Visualize chunk/section/doc coverage by plugin; show stale/dirty counts.

**Deliverables:** section-level JSON files, updated doc reducers, dependency tracker, dashboard coverage view.

### Phase C – Repository Hierarchies & Rollups (Weeks 6–9)

1. **Hierarchy Registry**
   - Define `hierarchies/<name>.json` describing nodes (id, parent, doc refs, weighting). Start with file tree and domain-based hierarchies.
   - Support custom hierarchies (importance, chronology) via config.
2. **Hierarchy Rollup Jobs**
   - Implement reducers that summarize each hierarchy node using children summaries (section/doc outputs). Store results under `summaries/<hierarchy>/<node>.json`.
   - Add metrics (child count, last refreshed).
3. **Dashboard & API**
   - Add hierarchy viewer (tree + node summary), node-level re-run controls, freshness indicators.
   - Expose `/api/hierarchy/<name>` for agents and exports.

**Deliverables:** hierarchy definitions, rollup reducers, dashboard hierarchy UI, API endpoints.

### Phase D – Retrieval & Export Enhancements (Weeks 10–12)

1. **Hierarchy-aware Retrieval**
   - Update retriever to fetch chunk + section summaries; support hybrid scoring (chunk similarity + section importance).
   - Introduce reranker (optional small cross-encoder or heuristics) for final ordering.
2. **Graph Integration**
   - Use chunk/entity metadata to populate graph with richer nodes/edges; link graph communities to hierarchy nodes.
   - Summaries of communities use hierarchy rollups as context.
3. **Exports**
   - Provide exports by hierarchy node (e.g., doc pack, domain digest).
   - Include citations and provenance (chunk IDs, section titles) in output.

**Deliverables:** enhanced retriever/exporter, updated graph generation, new dashboard options for exports.

### Phase E – Advanced Automation (Weeks 13–16)

1. **Scheduler / Orchestrator**
   - Configure recurring jobs (daily chunk refresh, weekly hierarchy rebuild) via CLI or dashboard scheduler.
2. **Agent Planner Hooks**
   - Provide high-level API summarizing current state (coverage, hierarchy nodes) for future agent workflows.
3. **Telemetry & Tuning**
   - Capture LLM token usage, latency, failure rates; log to `.knowledge/logs/pipeline_metrics.jsonl` and chart in dashboard.
4. **Future Prep**
   - Document integration paths for vLLM/Ollama, advanced PII detection, user profile personalization.

**Deliverables:** automation scripts, metrics logging, long-term integration guides.

## 5. Risk & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| LM Studio throughput constraints | Slow map-reduce runs | Enforce chunk-first strategy, throttle concurrency, allow optional remote LLM fallback |
| Hierarchy metadata bloat | Increased storage & rebuild time | Compress JSON (gzip), store diffs, prune obsolete nodes |
| Plugin rewrite complexity | Regression risk | Introduce unit tests and golden docs per plugin, stage rollout feature flags |
| Dashboard scope creep | UX clutter | Keep hierarchy views collapsible, prioritize observability over heavy interactivity |
| Data reprocessing cost | Long rebuild cycles | Use dependency graph to limit reprocessing, schedule heavier tasks off-hours |

## 6. Success Metrics

- **Coverage:** % docs with chunk summaries, section summaries, doc reduces, hierarchy rollups.
- **Latency:** average runtime per chunk summary, section reduce, doc reduce.
- **Token savings:** measured reduction in average tokens per doc-level attribute after hierarchy adoption.
- **Retrieval quality:** qualitative evaluation (user satisfaction) plus quantitative (hit rate on known questions).
- **Dashboard adoption:** frequency of map-reduce runs, hierarchy views, and per-attribute reruns.

## 7. Immediate Next Steps

1. Confirm time allocations and resource availability for Phase A (chunking + automation).
2. Create engineering tickets aligned with Phase A deliverables.
3. Establish test document set (large manuals, mixed media) to validate hierarchical workflows.
4. Instrument baseline metrics (current chunk coverage, average summary runtime) for before/after comparisons.

---

This roadmap keeps the current system stable while delivering the hierarchical intelligence envisioned in DESIGN.md and DESIGN_additions.md. Each phase produces user-visible value and prepares the ground for long-term ambitions such as agent planners and multi-perspective knowledge views.

