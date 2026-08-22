# AGSi-eval status

**Updated:** 2026-08-22 (council review)  
**Rank:** READY evaluation engineering · not ACTIVE science

| Subject | State | Claim tier |
|---------|--------|------------|
| R | Bound — B.0 keyword/fixture | P1 engineering |
| RG echo | Smoke only — circular with R | not a combined test |
| RG file wrap | Bound — `wrap_items.json` | P1 wrap-offline |
| G live | NOT_BOUND | P0 |
| Combined AGSi | SURMISE | — |
| Slice B.1 multi-turn | **Not built** | — |
| Slice A | Specified, not proof-grade runnable | — |

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject R --items science/agsi-eval/slice_b/items.json
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --adapter file:science/agsi-eval/slice_b/CANDIDATES.example.json --items science/agsi-eval/slice_b/wrap_items.json
```

Review: [`REVIEW_2026-08-22.md`](REVIEW_2026-08-22.md)
