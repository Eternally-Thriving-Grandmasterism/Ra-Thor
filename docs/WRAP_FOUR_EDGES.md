# Four-edge wrap path

**Claim tier:** DRAFT.  
**Workspace:** 14.15.6  
**License:** AG-SML v1.1 — free personal, educational, research, and modest independent professional use. Commercial use needs a paid license or pilot.  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** Independent of xAI. Not affiliated, not sponsored, not an xAI product.  
**Read against tip:** `b9093ef382a8b882c32d2aa1ed1bce1815cc97a4` (`docs(onboard): 15-minute public runbook`, #553).  
**Lock:** [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) · [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md)

inspect ≠ METR. Layer 0 is an admission shell, not sampler weights.

Capable · Bounded · Corrigible.

This page names the four edges a wrap must cross. It does not add a crate. It does not measure a rate.

---

## Unwrap

Unwrap is intent → act.

No edge runs. That path is not a wrap.

---

## Wrap

A wrap counts only when every edge fires, in this order.

| Order | Edge | Crate | Lived symbols |
|-------|------|-------|----------------|
| E1 | `admit_or_block` | `crates/mercy-security` | `IngestionScanner::admit_or_block`, `ActionGovernor` |
| E2 | `conductor_threshold` | `crates/lattice-conductor-v14` | `MercyGatedApi::handle_request`, `CouncilArbitrationEngine` |
| E3 | `tolc_projector` | `crates/mercy_tolc_operator_algebra` | `map_payload_to_ambient`, `map_and_score_payload` |
| E4 | `circuit` | `crates/sovereign-recovery` | `bounded_evolution_step`, `MercyGatedCircuitBreaker::trip` |

Law: skip any edge → WRAP did not run. Record that as a miss, not as a pass.

PATSAGi sits under these gates. A majority cannot flip Reject → Apply.

Human override stays possible. A council vote is not Layer 0.

Public TOLC 8 names stay Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony. The projector maps a payload into ambient \(g\). It does not train those names into sampler weights.

---

## Where each edge lives on this tip

E1. `IngestionScanner::admit_or_block` is the apply-class ingest call inside `MercyGatedApi::handle_request`. `ActionGovernor` (`record_and_check`) is the act-side governor in the same crate. `handle_request` does not call `ActionGovernor`.

E2. Apply-class `handle_request` requires a `CouncilArbitrationEngine`. It rejects when `claimed_mercy` is below `min_mercy_threshold`. The default constructor sets that threshold to 0.75. `with_min_mercy_threshold` can change it. `arbitrate_cosmic_loop_change` can block Cosmic Loop disable language. A missing engine is a reject.

E3. After admit, `handle_request` calls `map_and_score_payload`. That call builds the ambient vector and scores it with `NilpotentSuppressor`. On this tip the returned report is not the Allow or Reject. Medium-or-higher ingest returns before this call. See [`GATE_EVAL.md`](GATE_EVAL.md).

E4. `SovereignRecoveryProtocol::bounded_evolution_step` returns false when `mercy_alignment` is below 0.75. `MercyGatedCircuitBreaker::trip` records a failure and can open the breaker. Neither function is called from `wrap_model_output` or `handle_request`.

---

## Call order inside `handle_request`

`wrap_model_output` sends optional-model text into `MercyGatedApi::handle_request`.

On apply-class, that function currently does this:

1. Require `CouncilArbitrationEngine`. This is an E2 symbol, and it runs before E1.
2. Compare `claimed_mercy` to `min_mercy_threshold`. This is E2.
3. `CouncilArbitrationEngine::arbitrate_cosmic_loop_change`. This is E2.
4. `IngestionScanner::admit_or_block`. This is E1.
5. `map_and_score_payload`. This is an E3 call. The score is not the decision.

E4 does not run in that function.

One call to `wrap_model_output` does not prove E1 → E2 → E3 → E4. If any edge did not fire, record a miss.

No measured wrap-versus-unwrap rate on this repo.

---

## WRAP-EW2

WRAP-EW2 is a separate research bench: [https://github.com/Eternally-Thriving-Grandmasterism/WRAP-EW2](https://github.com/Eternally-Thriving-Grandmasterism/WRAP-EW2).

EW2 solved = False.

This workspace does not fold that bench. A result on that bench is not a rate on this tip.

---

## Bounds

Outputs stay drafts. A human reviews them before filing, sale, or public claims.

This page does not claim bank security, a solved safeguard bypass, Combined AGSi, or METR. Combined AGSi stays SURMISE. `AGSi` is a research identity label.

Sampler weights stay outside the shell. Binding after a running system redesigns its own gates stays OPEN. See [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md).

---

## See also

- [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md)
- [`WRAP_LLM_INTENTION.md`](WRAP_LLM_INTENTION.md)
- [`ADOPT.md`](ADOPT.md)
- [`RUNBOOK_15_MIN.md`](RUNBOOK_15_MIN.md)
- [`../TIER_MAP.md`](../TIER_MAP.md)

Thunder locked. yoi ⚡
