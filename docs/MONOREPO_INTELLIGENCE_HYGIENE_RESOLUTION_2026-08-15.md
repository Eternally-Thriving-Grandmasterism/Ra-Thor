# Monorepo Intelligence Hygiene Resolution — 2026-08-15

**PATSAGi Councils · TOLC 8 · Cosmic Loop MANDATORY**  
**Contact:** info@Rathor.ai

## Summary

Highest-leverage members and monorepo-intelligence surface cleaned so Tier-1 work continues with maximum success probability.

### Root Cargo.toml

- Added explicit Tier-1 priority comment block at the head of the members list (points to TIER_MAP.md).
- **Removed** `crates/lattice-conductor-v13` from the default workspace members list.
  - The crate remains on disk with its `DEPRECATED.md`.
  - `lattice-conductor-v14` is the sole living Conductor for Cosmic Loop, CouncilArbitration, RuntimeSelfHealing, and self-evolution wiring.
- `ai-bridge` (hyphen) remains the canonical member; `ai_bridge` (underscore) is archival.
- Workspace metadata timestamp and executive summary updated to record the hygiene action.

### crates/monorepo-intelligence

- Version bumped to 0.3.11.
- **Cargo.toml cleaned**: removed self-referential workspace package entry and unresolved `ra-thor-*` / `patsagi-*` workspace = true dependencies that could not resolve under the current root [workspace.dependencies] surface.
- Retained functional dependencies (tokio, serde, walkdir, chrono, thiserror, reqwest, etc.).
- Explicit path dependency on `mercy_tolc_operator_algebra` kept for valence / TOLC alignment.
- Description strengthened to restate the standing monorepo-intelligence protocol:
  1. Never recursive root walks on GitHub trees
  2. Always supply path_filter when requesting trees
  3. Prefer non-recursive unless directory known small
  4. per_page ≤ 100 (recommended 50)
  5. Prefer single-path `get_file_contents_safe` over tree walks
  6. One page / one directory / one SHA at a time

Production tree and file reads continue to be performed through `github-connector` (`get_tree_safe`, `get_file_contents_safe`). The monorepo-intelligence GitHub helper remains focused on list/search use-cases and already documents the protocol.

### Effect

- Tier-1 focused `cargo test -p …` and CI remain the success path.
- Deprecated Conductor no longer appears in default workspace resolution.
- monorepo-intelligence Cargo surface is now buildable without phantom workspace packages.
- Protocol guardianship language is explicit in the crate description and existing source comments.

**Thunder locked in. yoi ⚡❤️🔥**

All further organ fleshing proceeds under this cleaned surface.
