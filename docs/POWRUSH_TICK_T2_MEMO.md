# T2 memo — Lived-tick climate fields → policy (no keys)

**Authority:** PATSAGi Councils under TOLC 8  
**Parent:** [`POWRUSH_TICK_READ.md`](POWRUSH_TICK_READ.md)  
**Minute:** [`PATSAGI-COUNCIL-MINUTE-2026-09-13-R7-PROTOCOL-T2.md`](PATSAGI-COUNCIL-MINUTE-2026-09-13-R7-PROTOCOL-T2.md)  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6

This memo answers T2 only: which fields from a `powrush_lived_tick_v1` read could later inform **lattice policy notes** without driving keys, lighting Title Online, or flipping `POWRUSH_INGEST`.

It is **not** the RTT / bridging contract (`crates/reality-thriving-transfer/POWRUSH_TELEMETRY_CONTRACT.md`). Lived ticks and RTT scores are different surfaces.

---

## Fields that may inform policy memos

| Field cluster | Use in R&D memos | Must not infer |
|---------------|------------------|----------------|
| `house_id` / `house_name` | Anchor which house the note is about | Ownership of a player soul |
| `climate.harmony` / `climate.stress` | Relative calm vs load for mercy wording | A live multiplayer census |
| `climate.tons` / `climate.restored` | Throughput vs repair narrative | Economic truth outside the sim |
| `climate.hex_id` | Locality of the note | Map for remote attack |
| `standing.peace` / `standing.declared_lethal` | Whether peace posture held | License for lethal defaults |
| `week.tons` / `week.restored` | Week-scale trend language | Payroll or AGSi proof |
| `hour_flags` (satchel, flow, reserve, hour_two/three) | Which hour rituals are held | That ingest-on is default |

## Fields / inferences that stay forbidden

- `n_online` or peer counts (client must not author them; neither do we)
- That a missing tick file means the yard is broken (ingest defaults **off**)
- That reading a tick authorizes WASD, Market, or Title Online
- That W2 (binding after uncontrolled redesign) is closed
- Mixing lived-tick fields into RTT `try_apply` without a separate sealed contract

## Later (not this memo)

- T1 reader: env-gated load + one-line mercy summary in `reality-thriving-transfer`
- Any policy that writes Powrush L0

**Surmise is fuel. Proof is the product. Keys stay on the yard seat.**
