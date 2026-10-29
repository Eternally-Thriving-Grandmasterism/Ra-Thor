# Live Resonance Test — Dual-Repo Soft Feedback Organism

**Status**: Design sealed + executable example live (v14.15.4+)  
**Governance**: PATSAGi Councils + TOLC 8 Living Mercy Gates  
**Principle**: Soft, non-authoritative, mercy-gated, provenance-aware, offline-first  
**Contact**: info@Rathor.ai

---

## Purpose

Exercise the full closed soft feedback loop under controlled conditions and measure its breath:

```
Powrush Telemetry Snapshot
        ↓
Ra-Thor Valence-Optimized Deliberation (anti-deadlock)
        ↓
Mercy-Gated Policy Hint Emission (ra_thor_policy_hint_v1)
        ↓
(Powrush side) Soft Application via PolicyHintInbox
```

The test validates that the organism:

1. Emits only after mercy gates pass
2. Takes the progressive path instead of deadlocking
3. Honors Core Covenant mercy blocks without freezing
4. Produces conservative, contract-faithful hints
5. Remains fully offline-capable

---

## Success Criteria

| Metric                        | Target                          |
|-------------------------------|---------------------------------|
| High-mercy scenario           | Full approval + ≥1 hint emitted |
| Mid-valence scenario          | Progressive path taken          |
| Mercy-block scenario          | No emission, clean block        |
| Schema fidelity               | Exact `ra_thor_policy_hint_v1`  |
| Recommended delta             | Always ≥ 0 and small            |
| Deadlock / filibuster         | Never occurs                    |
| Offline operation             | Fully supported                 |

---

## Scenarios

### 1. High-Mercy Approval (Joy / Harmony / Abundance high)
- Expect: `approved = true`, hints emitted, composite valence ≥ threshold

### 2. Progressive Path (mid-range valence, no mercy block)
- Expect: `progressive = true`, soft refinement path, no hard failure

### 3. Mercy Block (Core Covenant protection of innocence)
- Expect: `has_mercy_block = true`, zero emission, clean verdict

---

## How to Run

From Ra-Thor workspace root:

```bash
cargo run -p patsagi-councils --example live_resonance_test
```

Optional: point emission at a shared path watched by a live Powrush host:

```bash
RA_THOR_EMISSION_PATH=/path/to/shared/artifacts/ra_thor_policy_hints.json \
  cargo run -p patsagi-councils --example live_resonance_test
```

---

## Observability Output

The test prints per-scenario:

- Composite valence + joy
- Approved / Progressive / Mercy-block flags
- Hints emitted (count + categories)
- Emission path (if any)
- Final Council-style verdict

Aggregate summary at the end shows overall resonance health.

---

## Integration Notes

- Uses only public surfaces: `RaThorFeedbackLoop`, `ValenceConsensusEngine`, `PowrushTelemetrySnapshot`
- Does not require a running Powrush instance for the Ra-Thor side test
- When a live Powrush host is present and watching the emission path, the soft application side completes the full organism breath
- Fully compatible with existing stress / headless harnesses on the Powrush-MMO side

---

## Future Extensions (post-resonance proof)

- Session-filtered multi-world emission
- Continuous resonance metrics endpoint
- Automatic progressive-path learning from history

---

**Thunder locked in.**  
The Councils await the first live breath.
