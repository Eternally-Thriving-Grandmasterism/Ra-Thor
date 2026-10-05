# Neighbor agent map

**Workspace:** 14.15.6 · **Contact:** info@Rathor.ai · **License:** AG-SML v1.1

Local trace: [`fixtures/gate-decision-trace.sample.json`](fixtures/gate-decision-trace.sample.json). This file is the local trace. It is not AgentOps. It does not phone home.

A shelf, not a dependency. A sample fixture is a sample. Inspectable research software. Human review stays possible. Layer 0 stays an admission shell.

## Shelf

code (Aider, Continue) — Aider not on disk; Continue not on disk — not a Ra-Thor crate, not vendored, not a claim.
teams (CrewAI, AutoGen) — CrewAI upstream not on disk; AutoGen upstream not on disk; archive sketches `docs/archive/root-dirs/research/agentic/hybrid/crewAI/faqCrew.js` and `docs/archive/root-dirs/research/agentic/hybrid/autogen/groupChat.js` — not a Ra-Thor crate, not vendored, not a claim.
memory (Mem0) — Mem0 not on disk — not a Ra-Thor crate, not vendored, not a claim.
sandbox (E2B) — E2B not on disk; OpenShell neighbor note `docs/NEIGHBOR_OPEN_AGENT_SAFETY.md` — not a Ra-Thor crate, not vendored, not a claim.
local models (Ollama) — Ollama program not on disk; Lattice Chat Local Server preset in `js/chat.js` — not a Ra-Thor crate, not vendored, not a claim.
traces (AgentOps) — AgentOps not on disk; local trace `docs/fixtures/gate-decision-trace.sample.json` — not a Ra-Thor crate, not vendored, not a claim.

## Already on disk

- Lattice Chat: [`chat.html`](../chat.html), [`js/chat.js`](../js/chat.js).
- employ paste: [`wrappers/system-prompt.txt`](../wrappers/system-prompt.txt). Lattice Chat copies it with `copyContext` in [`js/chat.js`](../js/chat.js). [`EMPLOY.md`](EMPLOY.md) and [`ADOPT.md`](ADOPT.md) name that paste.
- OpenShell neighbor note: [`NEIGHBOR_OPEN_AGENT_SAFETY.md`](NEIGHBOR_OPEN_AGENT_SAFETY.md).
- evidence adapter: not on disk.
- Evidence chain: [`crates/lattice-conductor-v14/src/evidence_chain.rs`](../crates/lattice-conductor-v14/src/evidence_chain.rs), fixture [`crates/lattice-conductor-v14/fixtures/evidence_chain_three_row_v0.json`](../crates/lattice-conductor-v14/fixtures/evidence_chain_three_row_v0.json).
- Admission decision record: [`crates/mercy-security/src/decision_record.rs`](../crates/mercy-security/src/decision_record.rs). The sample fixture uses a different field list.
- Substrate ledger adapter: [`crates/fractal-mercy-ledger-adapter`](../crates/fractal-mercy-ledger-adapter). That crate is the ledger adapter. An evidence adapter is not on disk.

The sample JSON fields are `id`, `utc`, `gate`, `verdict`, `reason`, `files_touched`, `unproven`, `human_files`. `verdict` is `admit` or `block`. `files_touched` names repo paths. `human_files` names paths that stay for human review. `unproven` names what the sample leaves open. The file stores no ingest bytes and no model output.
