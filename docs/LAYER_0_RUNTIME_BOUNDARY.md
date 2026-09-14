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

Controlled Cosmic Loop + human steward + Core Tier-1 + these edges: **bound in this repo** (R6 test).  
Binding after the running system redesigns its own gates: **OPEN** — see [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md). Do not close that gap in this file.

---

## Four runtime edges

| # | Boundary | Crate | Call site | Can refuse | Cannot refuse |
|---|----------|-------|-----------|------------|---------------|
| 1 | Geometry | `mercy_tolc_operator_algebra` | `map_payload_to_ambient`, `map_and_score_payload`, `NilpotentSuppressor::suppress_weighted`, zone `process` / `purify` | A lattice vector \(g \in \mathbb{R}^{16}\) off the 8-gate frame. Residual \(N_1(g)=(I-P_\lambda)g\). | The next token of a wrapped model, unless that token is mapped into \(g\). |
| 2 | Orchestration | `lattice-conductor-v14` | `MercyGatedApi::handle_request` + `CouncilArbitrationEngine` | Apply-class with no engine. `claimed_mercy` below `0.75`. Cosmic Loop disable language. | Process memory you already own. Cryptographic consensus. |
| 3 | Ingest / act | `mercy-security` | `admit_or_block` **wired into apply-class** | Unattended ingest at Medium+. Oversized payload. Credential / pickle / `trust_remote_code` class. | A model that never crosses this scanner. |
| 4 | Recover | `sovereign-recovery` | `bounded_evolution_step`, `MercyGatedCircuitBreaker` | Evolution step with `mercy_alignment < 0.75`. Open breaker. | A path that never calls the protocol. |

PATSAGi deliberates **under** those edges. A majority cannot flip a Rejected gate into apply. R6 asserts the API reject stands after a council review call.

---

## Apply-class vs read-class (edge 2 + 3 + map)

| Kind | Class | Missing arb | admit_or_block | map → g |
|------|-------|-------------|----------------|---------|
| `HealthCheck` | read | Allowed | skipped | skipped |
| `CosmicLoopStatus` | read | Allowed | skipped | skipped |
| `SubmitHealingIntent` | apply | **Rejected** | required | required |
| `CouncilQuery` | apply | **Rejected** | required | required |
| `SelfEvolutionProposal` | apply | **Rejected** | required | required |
| `Custom(_)` | apply | **Rejected** | required | required |

`map_payload_to_ambient` is a **deterministic feature map**, not an embedding model. Mapping runs before accept. Ingest refuse is the hard deny. Grief score is recorded by the map; it does not by itself reject a benign payload.

Wired entry `LatticeConductorV14::handle_mercy_api_request` always supplies the engine.

---

## Motions landed 2026-09-14

1. Map optional-model / apply payload into ambient \(g\) — `payload_map.rs`.
2. Fail-closed ingest on apply-class — `IngestionScanner::admit_or_block` inside `handle_request`.
3. R6 fail-closed test on the **controlled** loop — `tests/r6_fail_closed_controlled_loop.rs`.

Still open (not this repo's warranty):

- Sampler / weight constraint (not claimed).
- [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) — uncontrolled redesign.
- Hypervisor / seccomp / Lean kernel as default product.

---

## Non-claims

- Attached weights are not mercy-aligned because this repo exists.
- Keyword arbitration is a tripwire. It is not a proof about the next token.
- Valence `v ≥ 0.999999` is lattice law on mapped state.
- AGSi is a research identity label, not a warranty.

```bash
cargo test -p mercy_tolc_operator_algebra
cargo test -p lattice-conductor-v14
cargo test -p mercy-security
cargo test -p sovereign-recovery
```

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
