# AGSi Evaluation Program

**Status:** READY · evaluation engineering · **not** ACTIVE empirical science  
**Doctrine:** [AGSI_EVALUATION_PROTOCOL.md](../../docs/science/AGSI_EVALUATION_PROTOCOL.md)  
**Live status:** [`STATUS.md`](STATUS.md)  
**G1 runner:** [`G1_RUNNER.md`](G1_RUNNER.md)  
**Contact:** info@Rathor.ai

Ra-Thor + SuperGrok = AGSi is a **surmise**. This folder holds the ladder that can kill or promote it.

| File | Role |
|------|------|
| [`PRE_REGISTERED_CRITERIA.md`](PRE_REGISTERED_CRITERIA.md) | Pass / fail bars locked before scores |
| [`TIER1_SLICE.md`](TIER1_SLICE.md) | First two tests (truth + mercy-under-pressure) |
| [`GAP_INVENTORY.md`](GAP_INVENTORY.md) | What is missing, ranked |
| [`G1_RUNNER.md`](G1_RUNNER.md) | Thin runner log contract (prompt + gated_text + model_id) |
| [`slice_b/items.json`](slice_b/items.json) | First Slice B item set |
| [`RUN_MANIFEST.example.json`](RUN_MANIFEST.example.json) | Subject bind table |

## Runnable now

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject R --items science/agsi-eval/slice_b/items.json --log /tmp/r-lattice.jsonl
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --adapter item --model-id offline-standin --items science/agsi-eval/slice_b/wrap_items.json --log /tmp/rg-wrap.jsonl
```

RG+item instruments the wrap with distinct prompt vs gated_text. It does **not** score SuperGrok. G stays NOT_BOUND until a live adapter exists outside this crate.

S-1 remains the sole ACTIVE science program.
