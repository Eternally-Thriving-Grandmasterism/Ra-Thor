# POWRUSH_TICK_READ.md — lattice R&D from the yard tick

**Contact:** info@Rathor.ai. Independent of xAI. Not certified.
**Workspace:** 14.15.6. No 15.x. No `crates/self-evolution`.
**Surfaces:** Ra-Thor reads. Powrush plays. Lattice **never drives keys**.

Companion (other repo): `Powrush-MMO/docs/AAA_FEEL_WORKPACK.md`
Ingest flag lives in Powrush: `POWRUSH_INGEST` default **off**.
Path when on: `data/powrush_lived_tick.json` schema `powrush_lived_tick_v1`.

## What you may do with a tick

Read house_id, house_name, climate (harmony/stress/tons/restored/hex_id),
standing (peace, declared_lethal), week (tons + restored), hour_flags
(satchel, flow, reserve, hour_two/three held).

Use those fields for **policy notes and future R&D memos**. Do not write
Powrush L0. Do not emit WASD. Do not light Title Online from a tick.

## What you must not infer

- `n_online` or peer counts (client must not author them; neither do we)
- That ingest-on is the stranger default
- That a missing file means the yard is broken (flag is off)
- Binding after uncontrolled self-redesign is solved (W2 stays OPEN)

## R&D queue on this repo

| Door | Job |
|---|---|
| **R6** | One fail-closed Cosmic Loop test in one Tier-1 crate |
| **T1** | Optional reader that loads a tick JSON from a path env and prints a one-line mercy summary — **off unless path set**, no network |
| **T2** | Memo: which climate fields would later inform policy without touching keys |

T1 is not a product. Do not add workspace members for it unless steward names the crate.

## Dual-repo

Powrush Bot 1 owns feel slices A1–A5.
This seat owns R6 then T1/T2.
`POWRUSH_INGEST` is **never flipped on from this repo**.
