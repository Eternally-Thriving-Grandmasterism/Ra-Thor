# Independent Keeper Packet — Slice B

**Authority:** Permanent PATSAGi Councils under TOLC 8  
**Sealed by:** [`docs/science/PATSAGI-COUNCIL-MINUTE-2026-09-01-G3-KEEPER.md`](../../docs/science/PATSAGI-COUNCIL-MINUTE-2026-09-01-G3-KEEPER.md)  
**Claim tier of this file:** P0 packet · a completed receipt is P1 engineering · **not** P3  
**Contact:** info@Rathor.ai  
**Date:** 2026-09-01

This packet exists because a lattice self-score is not independence.

## Who may be a keeper

A keeper is a named person or lab that is **not** the sole steward and **not** the session that authored the packet.

Forbidden as keeper for a P2 claim: Autonomicity Games Inc. staff acting as first party; this Grok session; any party that edited `PRE_REGISTERED_CRITERIA.md` in the same week as the run.

## What you receive

| Object | Path |
| --- | --- |
| Bars | `science/agsi-eval/PRE_REGISTERED_CRITERIA.md` |
| Runner contract | `science/agsi-eval/G1_RUNNER.md` |
| B.0 items | `science/agsi-eval/slice_b/items.json` |
| B.0 wrap-distinct | `science/agsi-eval/slice_b/wrap_items.json` |
| B.1 multi-turn (25) | `science/agsi-eval/slice_b1/items.json` |
| Receipt schema | `science/agsi-eval/RECEIPT.example.json` |

Do not edit those files for the run. Pin the git SHA instead.

## Commands (do not invent a second harness)

From a clean clone of `Eternally-Thriving-Grandmasterism/Ra-Thor` at a recorded SHA:

```bash
git rev-parse HEAD

cargo test -p mercy-security agsi_eval

cargo run -p mercy-security --bin agsi-eval-rg -- \
  --subject R \
  --model-id lattice-native \
  --items science/agsi-eval/slice_b/items.json \
  --log keeper-r-b0.jsonl

cargo run -p mercy-security --bin agsi-eval-rg -- \
  --subject RG --adapter item \
  --model-id offline-standin \
  --items science/agsi-eval/slice_b/wrap_items.json \
  --log keeper-rg-wrap.jsonl

cargo run -p mercy-security --bin agsi-eval-rg -- \
  --slice b1 \
  --model-id lattice-native \
  --items science/agsi-eval/slice_b1/items.json \
  --log keeper-b1.jsonl
```

`--subject G` must remain NOT_BOUND. Do not attach a live model and call it this packet.

## What to return

Email **info@Rathor.ai** with:

1. Completed receipt JSON (see example).
2. The three JSONL logs.
3. `git rev-parse HEAD` and `rustc --version`.
4. One sentence of what you are **not** claiming.

## Language keepers may not use

- “We proved AGSi”
- “Non-bypassable under a named adversary” (that is the P3 bar; it needs B1/B2)
- “Independent of the lattice” if you modified gates for the run
- Any homepage valence / APTD score as the result of this packet

## What a returned receipt is

| If | Then |
| --- | --- |
| Receipt + logs, bars unchanged | P1 external engineering |
| Same, plus a second named lab | still not P3 until B1/B2 exist |
| Live G mixed into these commands | invalid packet — file a separate adapter note |

S-1 remains the sole ACTIVE empirical science program.
