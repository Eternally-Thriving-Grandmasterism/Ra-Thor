# G1 — thin runner log contract

**Date:** 2026-09-01  
**Status:** READY evaluation engineering · not ACTIVE science  
**Claim tier:** P1 instrumentation · not a live SuperGrok score · not P3  
**Contact:** info@Rathor.ai

G1 asked for one external-run entrypoint that logs **gate traces + model id + prompt**.  
That entrypoint is the existing binary:

```bash
cargo run -p mercy-security --bin agsi-eval-rg -- \
  --subject RG --adapter item \
  --model-id offline-standin \
  --items science/agsi-eval/slice_b/wrap_items.json \
  --log science/agsi-eval/logs/rg-wrap.example.jsonl
```

Subject R (lattice only):

```bash
cargo run -p mercy-security --bin agsi-eval-rg -- \
  --subject R \
  --model-id lattice-native \
  --items science/agsi-eval/slice_b/items.json \
  --log science/agsi-eval/logs/r-lattice.example.jsonl
```

Subject G remains **NOT_BOUND**. `--subject G` still prints the unbound report and writes no scored traces. Passing `--model-id grok-…` does not bind G.

## JSONL line (one gate decision)

| Field | Meaning |
| --- | --- |
| `item_id` | Item id from the slice file |
| `subject` | `R` or `RG` |
| `model_id` | Caller pin (`--model-id`) or adapter name |
| `adapter` | `none` / `echo-smoke` / `item-candidate` / `file` / `multi-turn-candidate` |
| `prompt` | User / item prompt |
| `gated_text` | Text the gates actually scored (candidate, not the prompt, on a real wrap) |
| `reason` | Gate reason string |
| `expected_block_or_refuse` / `observed_block_or_refuse` / `correct` | Engineering tally only |

## What this does not do

- Does not call SuperGrok / Grok / any live HTTP model.
- Does not score Slice A.
- Does not promote AGSi-eval to ACTIVE.
- Does not satisfy P3 (independent logs + named B1/B2 still missing).

S-1 remains the sole ACTIVE empirical program.
