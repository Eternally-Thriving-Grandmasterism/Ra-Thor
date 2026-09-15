# Layer 0 — runtime boundary (not a trained value)

**Date:** 2026-09-14  
**Workspace:** 14.15.6 (do not bump in this motion)  
**Trigger:** public question — corrigibility as a runtime invariant, not a preference in weights.  
**Repo:** https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor  
**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1  
**Status:** inspectable research software. Optional Grok session. Independent of xAI. Not certified. Not a legal product.

This card is law for agents and public readers. It does not enlarge product scope. It names where Layer 0 is enforced and where it is not.

Premise accepted: *Corrigibility only holds as an invariant enforced at the runtime boundary, not as a value trained into weights. Policy layers drift; kernel constraints don't.*

---

## One screen

Layer 0 is enforced on **lattice state and admission**. It is not enforced inside Grok, Claude, Gemini, Ollama, WebLLM, or any other sampler.

If a session never crosses the edges below, Layer 0 did not run.

Controlled Cosmic Loop + human steward + Core Tier-1 + these edges: **bound in this repo** (R6 + wrap tests).  
Binding after the running system redesigns its own gates: **OPEN** — see [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md). Do not close that gap in this file.

---

## Four runtime edges + wrap

| # | Boundary | Crate | Call site | Can refuse | Cannot refuse |
|---|----------|-------|-----------|------------|---------------|
| 1 | Geometry | `mercy_tolc_operator_algebra` | `map_payload_to_ambient`, `map_and_score_payload`, suppressor / `purify` | Mapped \(g \in \mathbb{R}^{16}\) off the 8-gate frame. | Tokens never mapped. |
| 2 | Orchestration | `lattice-conductor-v14` | `MercyGatedApi::handle_request` + `CouncilArbitrationEngine` | Missing engine. Low `claimed_mercy`. Cosmic Loop disable language. | Process memory you already own. |
| 3 | Ingest / act | `mercy-security` | `admit_or_block` on apply-class | Medium+ ingest, pickle / `trust_remote_code` class. | A model that never crosses this scanner. |
| 4 | Recover | `sovereign-recovery` | `bounded_evolution_step` | Low mercy alignment. | A path that never calls the protocol. |
| 5 | Wrap | `lattice-conductor-v14` | `wrap_model_output` / `LatticeConductorV14::wrap_model_output` | Same as 2+3 on optional-model text that uses this path. | A chat that pastes tokens into `main` without calling it. |

---

## Motions landed 2026-09-14

1. Map payload → \(g\).
2. Fail-closed ingest on apply-class.
3. R6 controlled-loop tests.
4. `wrap_model_output` — optional-model apply envelope.
5. `submit_self_evolution_proposal_securely` → `handle_request` (#475).
6. After #476: `GitHubSurface::queue_evolution_pr` calls that same apply-class shell (engine + admit + map → \(g\)). No third keyword list.

**STOP Layer 0** unless a real default-member apply-class bypass is found by search. Do not invent a ninth path.

Keyword inspect (not METR): [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md).

Still open:

- Sampler / weight constraint (not claimed).
- [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md).
- Hypervisor / seccomp / Lean kernel as default product.

```bash
cargo test -p mercy_tolc_operator_algebra
cargo test -p lattice-conductor-v14
cargo test -p mercy-security
cargo test -p ra-thor-one-organism
```

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
