# AGSi-eval status

**Updated:** 2026-08-22  
**Rank:** READY evaluation engineering · not ACTIVE science  
**Claim tier:** R = P1 lattice-only · RG+echo/file = P1 wrap-offline · G live = NOT_BOUND · Combined AGSi = SURMISE

## Runnable now

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject R --items science/agsi-eval/slice_b/items.json
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --adapter echo --items science/agsi-eval/slice_b/items.json
cargo run -p mercy-security --bin agsi-eval-rg -- --subject G --items science/agsi-eval/slice_b/items.json
```

`--subject G` must print NOT_BOUND. That is correct.
`--subject RG --adapter echo` scores the *wrap path* using a complying stand-in generator. It is **not** a SuperGrok score.

## Still open

| Item | State |
|------|-------|
| Live G / SuperGrok adapter | Not shipped (no API keys in-tree) |
| Independent party + B1/B2 | Not scheduled |
| Slice A TruthfulQA pin | Not started |
| S-1 First-5 labels | Separate ACTIVE door |
