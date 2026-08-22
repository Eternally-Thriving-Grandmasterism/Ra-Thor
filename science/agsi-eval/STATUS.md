# AGSi-eval status

**Updated:** 2026-08-22  
**Rank:** READY evaluation engineering · not ACTIVE science  
**Claim tier:** Subject R = P1 lattice-only · Subjects G/RG = P0 NOT_BOUND · Combined AGSi = still SURMISE

## Runnable now

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject R --items science/agsi-eval/slice_b/items.json
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --items science/agsi-eval/slice_b/items.json
```

`--subject G` / `--subject RG` emit a NOT_BOUND report (exit 0). That is correct: the combined claim is not yet testable.

## Still open

| Item | State |
|------|-------|
| Model adapter for G / RG | Missing |
| Independent party + B1/B2 | Not scheduled |
| Slice A TruthfulQA pin | Not started |
| S-1 First-5 labels | Separate ACTIVE door |
