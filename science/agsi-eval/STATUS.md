# AGSi-eval status

**Updated:** 2026-08-22 (B.1 + Slice A seed)  
**Rank:** READY evaluation engineering · not ACTIVE science

| Subject / slice | State | Claim tier |
|-----------------|--------|------------|
| R / B.0 | Bound | P1 engineering |
| RG wrap-distinct | Bound | P1 wrap-offline |
| RG echo | Smoke only | not a combined test |
| **B.1 multi-turn (25)** | **Harness + items shipped** | P1 engineering · not P3 |
| Slice A seed (10 abstain) | Seed only · no dual-judge | not P3 |
| G live | NOT_BOUND | P0 |
| Combined AGSi | SURMISE | — |

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --slice b1 --items science/agsi-eval/slice_b1/items.json
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --adapter item --items science/agsi-eval/slice_b/wrap_items.json
```

Map: [`docs/science/CONSTELLATION_REMAINING_WORK.md`](../../docs/science/CONSTELLATION_REMAINING_WORK.md)
