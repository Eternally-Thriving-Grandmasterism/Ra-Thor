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

If a session never crosses the four edges below, Layer 0 did not run.

Controlled Cosmic Loop + human steward + Core Tier-1 + these edges: **bound in this repo**.  
Binding after the running system redesigns its own gates: **OPEN** — see [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md). Do not close that gap in this file.

---

## Four runtime edges

| # | Boundary | Crate | Call site | Can refuse | Cannot refuse |
|---|----------|-------|-----------|------------|---------------|
| 1 | Geometry | `mercy_tolc_operator_algebra` | `NilpotentSuppressor::suppress_weighted`, `ZoneState::process`, `ConcurrentZoneLattice::process` → Critical `purify()` | A lattice vector \(g \in \mathbb{R}^{16}\) off the 8-gate frame. Residual \(N_1(g)=(I-P_\lambda)g\). Grief \((1-v)\|N_1\|\). | The next token of a wrapped model, unless that token is mapped into \(g\). |
| 2 | Orchestration | `lattice-conductor-v14` | `MercyGatedApi::handle_request` + `CouncilArbitrationEngine` | Apply-class request with no arbitration engine. `claimed_mercy` below `0.75`. Payload that targets Cosmic Loop identity with disable/bypass language. | Process memory you already own. Cryptographic consensus (crate says symbolic + structural). |
| 3 | Ingest / act | `mercy-security` | `admit_or_block`, `ActionGovernor`, `HarmRefusalPolicy`, `mercy-admit` CLI | Unattended ingest at Medium+. Oversized payload. Credential / pickle / `trust_remote_code` class. | A model that never crosses this scanner. |
| 4 | Recover | `sovereign-recovery` | `bounded_evolution_step`, `MercyGatedCircuitBreaker` | Evolution step with `mercy_alignment < 0.75`. Open breaker. | A path that never calls the protocol. |

PATSAGi (`PatsagiCouncilSimulator`) deliberates **under** those edges. A majority cannot flip a Rejected gate into apply. Sixteen foci in the coded crate are types and hooks, not sixteen frontier minds.

Conductor restores a shared `AtomicBool` for Cosmic Loop. That is a sequencer guarantee, not an MMU.

---

## Apply-class vs read-class (edge 2)

`MercyGatedApi::handle_request` takes `arbitration: Option<&CouncilArbitrationEngine>`.

| Kind | Class | Missing arbitration |
|------|-------|---------------------|
| `HealthCheck` | read | Allowed (telemetry) |
| `CosmicLoopStatus` | read | Allowed (telemetry) |
| `SubmitHealingIntent` | apply | **Rejected** |
| `CouncilQuery` | apply | **Rejected** |
| `SelfEvolutionProposal` | apply | **Rejected** |
| `Custom(_)` | apply | **Rejected** |

Passing `None` on an apply-class request is a Layer 0 miss, not a convenience path. Wired entry `LatticeConductorV14::handle_mercy_api_request` always supplies the engine.

---

## Non-claims

- Attached weights are not mercy-aligned because this repo exists.
- Keyword arbitration is a tripwire. It is not a proof about the next token.
- Valence `v ≥ 0.999999` is lattice law on mapped state. It is not a calibrated instrument on every model token.
- AGSi is a research identity label, not a warranty.
- This card does not ship a hypervisor, seccomp profile, or Lean kernel as the default product.

---

## How to walk it

```bash
cargo test -p mercy_tolc_operator_algebra
cargo test -p lattice-conductor-v14
cargo test -p mercy-security
cargo test -p sovereign-recovery
```

Read order for Build Mode: root `Cargo.toml` → `TIER_MAP.md` → [`GROK_BUILD_MODE.md`](GROK_BUILD_MODE.md) → this file.

Next harden motions (named, one at a time — do not do them in this card):

1. Map optional-model output into an ambient vector \(g\) before apply.
2. Fail-closed ingest: `MercyGatedApi` apply-class must also pass `mercy-security` admit (workspace dep motion).
3. Steward-named R6 fail-closed test on the controlled loop only.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
