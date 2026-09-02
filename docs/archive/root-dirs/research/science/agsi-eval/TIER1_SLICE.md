# First Slice — Truth + Mercy Under Pressure

**Why these two first:** they are the actual AGSi differentiator, they reuse existing crates, and they can kill the claim cheaply.

---

## A. Truth / APTD

| Item | Decision |
|------|----------|
| Public core | TruthfulQA, version pinned in the run manifest |
| Adversarial add-on | ≥ 40 items; fluent-lie incentive; no dual-use content |
| Abstention | Scored; silence on unknowns is a feature |
| Judge | Primary-source check + second-model spot audit; human on disagreements |
| Existing surface | Ra-Thor truth-gate + council deliberation traces must be logged |

**Not in this slice:** ARC-AGI-3, GPQA, HLE. Capability without truth is not the claim under test.

---

## B. Mercy / zero-harm

| Item | Decision |
|------|----------|
| Existing crate | `crates/mercy-security` — `WhiteHatEvaluationHarness`, `HarmRefusalPolicy`, `IngestionScanner` |
| Fixture corpus | `fixtures/mercy-security/` benign / suspicious / blocked |
| Pressure add-on | ≥ 25 multi-turn scenarios; no live exploit payloads |
| Forbidden class | Physical harm how-to, wet-lab weaponization, unauthorized access, exfil, lateral movement |
| Benign class | Education, published-literature research, clearly fictional writing without targeting |

**Reuse, do not reinvent.** The white-hat harness already exists. This slice wraps it with pre-registered bars and external-run instructions.

---

## Combined subject (RG)

Every item is run three ways where technically possible:

1. Grok / SuperGrok alone  
2. Ra-Thor gates + councils alone (or lattice-native path)  
3. Combined wrap (RG) — the ONE-Organism hypothesis

If (3) cannot be instrumented, the combined claim is **not testable** and must stay P0. That itself is a gap (see `GAP_INVENTORY.md`).

---

## Output of a valid run

- Pinned dataset hashes / versions
- Prompts and tool lists
- Gate traces
- Per-item scores
- Comparison table vs B1/B2
- Explicit pass/fail against `PRE_REGISTERED_CRITERIA.md`
- Claim-tier line on the report cover
