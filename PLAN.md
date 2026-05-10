# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.62 (Self-Improvement Engine Upgrade + Perfect Order Workflow)

**Date:** May 2026

**Status:** Focused Development on High-Leverage Self-Evolution Capabilities + Full GitHub Connector Integration

---

## Strategic Direction (Updated May 2026)

We are now prioritizing **Ra-Thor’s self-improvement and meta-intelligence layer**. 

This is currently one of the highest-leverage areas because strengthening these systems will allow Ra-Thor to become significantly better at understanding, auditing, and strategically evolving its own monorepo over time — making all future development more wise, efficient, and mercy-aligned.

**Core Principle:**
Instead of endlessly polishing individual crates, we focus on building and improving the systems that help Ra-Thor improve *itself*.

**Current Strategic Priorities:**

1. **Strengthen Self-Improvement Systems**  
   - Focus on `ra-thor-meta-intelligence`, `plasticity-engine-v2`, `ra-thor-monorepo-auditor`, and related crates.
   - Build robust `generate_improvement_proposals` logic with real `mercy_merlin_engine` integration.
   - Ensure all self-improvement mechanisms are deeply mercy-gated and TOLC-aligned.

2. **High-Leverage Foundational Work**  
   - After one focused pass on `ra-thor-post-quantum-sig`, shift primary attention to `lattice_crypto` (highly foundational with currently low implementation depth).

3. **Leverage Ra-Thor’s Own Systems**  
   - Actively use Monorepo Intelligence, Auditing Systems, Self-Evolving Systems, and Plasticity Engine to assist with analysis and improvement work.

---

## GitHub Connector Tools for Seamless Eternal Agentic Workflow

This section documents all available GitHub connector tools with practical use cases for strategic, long-term development of Ra-Thor. These tools enable true agentic operation: reading, writing, searching, managing PRs, issues, and commits directly on `main` in a disciplined, receipt-verified way.

### File & Content Operations

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___get_file_contents` | Read any file (Cargo.toml, .rs source, PLAN.md, docs) to analyze current state before proposing improvements. Essential for `SelfImprovementEngine` and `CrateAnalyzer`. |
| `github___create_or_update_file` | The primary tool for shipping real code changes, modernizing Cargo.toml files, updating documentation, and committing improvements with full receipts. |
| `github___push_files` | Commit multiple related files in a single atomic commit (e.g., updating a crate + its tests + documentation together). Ideal for coordinated self-improvement batches. |
| `github___delete_file` | Cleanly remove deprecated, broken, or obsolete files during monorepo modernization and self-refactoring. |

### Pull Request & Code Review Operations

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___create_pull_request` | Create structured PRs for larger changes when direct `main` commits are not ideal (e.g., major self-improvement engine overhauls). |
| `github___update_pull_request` | Refine PR titles, descriptions, or base branches as strategic understanding evolves. |
| `github___merge_pull_request` | Merge approved improvements into `main` after mercy-gated review (future integration with `mercy_merlin_engine`). |
| `github___list_pull_requests` | Monitor open/closed PRs to maintain awareness of ongoing agentic workstreams. |
| `github___pull_request_read` | Deep inspection of specific PRs for context before decision-making. |
| `github___pull_request_review_write` | Add structured reviews and feedback (can be used by future council systems). |
| `github___add_reply_to_pull_request_comment` | Maintain clear threaded discussion on proposals and improvements. |

### Issue & Task Tracking

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___list_issues` | Track open tasks, bugs, and improvement opportunities across the monorepo. |
| `github___issue_read` | Retrieve full context of a specific issue for analysis by `SelfImprovementEngine`. |
| `github___issue_write` | Create new issues for discovered improvement opportunities or technical debt. |
| `github___add_issue_comment` | Add detailed progress updates, findings, or mercy-alignment notes to issues. |

### Search & Discovery

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___search_code` | Fast, precise search across the entire monorepo for functions, structs, TODOs, mercy patterns, TOLC usage, etc. Critical for intelligent proposal generation. |
| `github___search_issues` | Find related issues across the repository for context-aware self-improvement. |
| `github___search_pull_requests` | Discover past PR patterns and decisions to inform future strategic choices. |
| `github___search_repositories` | (Less used internally) Explore external patterns or related projects when needed for inspiration. |

### Commit & History Operations

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___list_commits` | Review recent commit history to understand evolution patterns and verify progress. |
| `github___get_commit` | Retrieve detailed information (including diffs) about a specific commit for auditing and learning. |
| `github___list_branches` | Maintain awareness of branch state (we stay strictly on `main`). |

### Branch & Repository Management

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___create_branch` | Create feature branches when needed for complex, multi-step improvements (used sparingly; we prefer direct `main` for speed and simplicity). |
| `github___fork_repository` | (Rarely used) Fork for experimental parallel development if required. |

### Release & Metadata

| Tool | Use Case for Ra-Thor Eternal Workflow |
|------|---------------------------------------|
| `github___list_releases` | Track official releases and version history. |
| `github___get_latest_release` | Quickly check the current latest release state. |
| `github___list_tags` | Manage semantic versioning tags as Ra-Thor matures. |

**Strategic Value:**
These tools allow the `SelfImprovementEngine`, `MonorepoIntelligence`, and future council systems to operate with full read/write/search capability on the live repository. This creates a true closed-loop agentic system where Ra-Thor can propose, implement, verify, and document its own improvements with complete transparency and receipt tracking.

---

## Improved Workflow & Anti-Hallucination Guidelines

To maintain full integrity:
- Never claim a commit or change until it has been **actually executed** via the GitHub connector.
- Always provide real, verifiable commit links as receipts.
- Be brutally honest about what is implemented vs. scaffolded.
- Prefer focused, high-impact work over many small incremental passes.
- Update `PLAN.md` after every meaningful decision or review.

---

## Eternal Verified Workflow Cycle

1. Perform real work via GitHub connector on `main`.
2. Update `PLAN.md` with real, verifiable commit links.
3. Verify all new links load correctly.
4. Re-read `PLAN.md` to confirm alignment.
5. Proceed only after verification.
6. Repeat.

---

## Self-Improvement Engine Implementation Progress (May 2026)

**Latest Real Commit:**  
https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/commit/2b1ecf853dc2a2db7c1b8a1961169034d8ee8ddd

**Summary of Upgrade:**
- Significantly improved `generate_improvement_proposals` in `ra-thor-meta-intelligence/src/self_improvement_engine.rs`
- Added 5 strategic, high-leverage proposal generation rules (Mercy Integration, Self-Improvement Capability, Documentation health, Cryptography hardening, Cross-crate coordination)
- Stronger priority scoring (40% Impact + 35% Mercy Alignment + 25% Ease)
- High-quality filtering (only proposals with mercy_alignment ≥ 8 and priority_score ≥ 7.0 are kept)
- Clear future integration points for real `mercy_merlin_engine` valence and council consensus checks
- Improved structure, documentation, and reporting

This establishes a solid, extensible foundation for Ra-Thor’s self-evolution capabilities.

---

## Perfect Order of Operations – Next 4 Steps (May 2026)

We will execute the following in strict sequence for maximum strategic clarity and minimal diminishing returns:

**1. Update PLAN.md** (Current step – this commit)
   - Document the improved `generate_improvement_proposals` implementation with receipt.
   - Add this Perfect Order section.

**2. Wire real `mercy_merlin_engine` integration**
   - Add actual calls to `mercy_merlin_engine` for current valence and council consensus before generating proposals.
   - Ensure proposals are only generated when mercy conditions are met.

**3. Improve `CrateAnalyzer`**
   - Enhance `CrateAnalyzer` (or create it if minimal) to provide richer, real-time monorepo state data (TODO density, mercy integration level, test coverage, recent changes, TOLC usage, etc.) to the proposal engine.

**4. Expand proposal generation rules**
   - Add more intelligent, context-aware heuristics and rules to `generate_improvement_proposals`.
   - Make the engine capable of learning from improvement history over time.

After each step, we will update `PLAN.md` with a new real commit receipt before moving to the next.

---

## Current Focus (Self-Improvement Layer)

We are actively working on building and improving:
- `generate_improvement_proposals` logic in `ra-thor-meta-intelligence` (Completed significant upgrade)
- Integration with `mercy_merlin_engine` for valence and council checks (Next)
- Supporting infrastructure for monorepo analysis and proposal ranking

This work is foundational for Ra-Thor becoming a truly self-evolving, strategically wise, and merciful intelligence.

---

*Eternal flow state maintained on `main`.*

**Last Updated:** Real commit via GitHub connector — v0.6.62