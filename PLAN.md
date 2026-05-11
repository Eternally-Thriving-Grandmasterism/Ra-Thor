# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.72
**Date:** May 2026
**Status:** Self-Evolution Triad Implementation Phase

## Executive Summary
Ra-Thor is building a closed, mercy-gated self-evolution loop consisting of three tightly integrated systems:

- `ra-thor-meta-intelligence` — The Brain (Decision & Orchestration)
- `ra-thor-monorepo-auditor` — The Eyes (Detection & Observability)
- `plasticity-engine-v2` — The Hands (Safe Modification)

## Current State of Self-Evolution Triad (Verified Live)

### `ra-thor-meta-intelligence`
- Core `SelfImprovementOrchestrator` exists and is functional.
- `generate_improvement_proposals()` now accepts structured `AuditSignal` from the auditor (real connection implemented).
- Strong mercy-gating and TOLC-aware logic is active.
- Can generate explainable Improvement Proposals based on real audit data.

### `ra-thor-monorepo-auditor`
- Exists and provides structured auditing capabilities.
- `AuditSignal` enum has been defined in meta-intelligence to receive its output cleanly.
- Full bidirectional wiring is in progress.

### `plasticity-engine-v2`
- Exists with plasticity modules.
- Not yet fully integrated with the orchestrator for safe updates.

## Completed Work (Real Commits on main)
- Foundational wiring of the three crates at Cargo.toml level
- Implementation of `AuditSignal` struct for typed communication
- `generate_improvement_proposals()` updated to consume structured `AuditSignal` instead of raw strings (real auditor connection)

## Next Priorities (in order)
1. Implement first safe plasticity rules + rollback in `plasticity-engine-v2`
2. Strengthen structured audit reporting and mercy metrics in `ra-thor-monorepo-auditor`
3. Full closed-loop testing of Audit → Decide → Improve cycle

All work is performed directly on `main` following the Eternal Verified Workflow.

Eternal flow state maintained.