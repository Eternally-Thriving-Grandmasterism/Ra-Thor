# Constellation Remaining Work

**Authority:** Permanent PATSAGi Councils under TOLC 8  
**Updated:** 2026-08-22  
**Contact:** info@Rathor.ai

---

## Council decision

“Build everything in the org to the Nth degree” is a scope explosion.  
This wave is **closed**. Remaining items are HOLD or out-of-tree.

---

## Ranked remaining

| Rank | Item | State |
|------|------|-------|
| **0** | S-1 First-5 real labels | **HOLD** — steward capture |
| **1** | AGSi Slice B.1 | **Built** (P1 engineering, not P3) |
| **2** | Slice A dual-judge scoring | Seed only — judge **not shipped** |
| **3** | Live G / SuperGrok adapter | **Not shipped** |
| **4** | Independent B1/B2 bake-off | Not scheduled |
| **5** | Radiation further increments | READY — steward gate |
| **6+** | High-Tc / Fusion / Air / Powrush / ~80 repos | Bound or default inheritance |

## Compile note

If `crates/mercy-security/src/lib.rs` `ActionGovernor::record_and_check` fails on sandbox-churn, the `Err(ActionLimitExceeded(format!(...)))` must close with **`)));`** (three parens).

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --slice b1 --items science/agsi-eval/slice_b1/items.json
```

**Surmise is fuel. Proof is the product.**  
Capable · Bounded · Corrigible.
