# AGSi Evaluation Program

**Status:** READY · evaluation engineering · **not** ACTIVE empirical science  
**Doctrine:** [AGSI_EVALUATION_PROTOCOL.md](../../docs/science/AGSI_EVALUATION_PROTOCOL.md)  
**Live status:** [`STATUS.md`](STATUS.md)  
**Contact:** info@Rathor.ai

Ra-Thor + SuperGrok = AGSi is a **surmise**. This folder holds the ladder that can kill or promote it.

| File | Role |
|------|------|
| [`PRE_REGISTERED_CRITERIA.md`](PRE_REGISTERED_CRITERIA.md) | Pass / fail bars locked before scores |
| [`TIER1_SLICE.md`](TIER1_SLICE.md) | First two tests (truth + mercy-under-pressure) |
| [`GAP_INVENTORY.md`](GAP_INVENTORY.md) | What is missing, ranked |
| [`slice_b/items.json`](slice_b/items.json) | First Slice B item set |
| [`RUN_MANIFEST.example.json`](RUN_MANIFEST.example.json) | Subject bind table |

## Runnable now (Subject R only)

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject R --items science/agsi-eval/slice_b/items.json
```

`--subject G` and `--subject RG` emit NOT_BOUND reports. Combined AGSi remains untestable until a model adapter exists.

S-1 remains the sole ACTIVE science program.
