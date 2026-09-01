# AGSi-eval status

**Updated:** 2026-09-01 (G3 keeper packet sealed)  
**Rank:** READY evaluation engineering · not ACTIVE science

| Subject / slice | State | Claim tier |
|-----------------|--------|------------|
| R / B.0 | Bound + traces | P1 engineering |
| RG wrap-distinct | Bound + traces | P1 wrap-offline |
| RG echo | Smoke only | not a combined test |
| **B.1 multi-turn (25)** | Harness + items + traces | P1 engineering · not P3 |
| Independent keeper packet | **Shipped** | P0 packet · receipt would be P1 |
| Slice A seed (10 abstain) | Seed only · no dual-judge | not P3 |
| G live | NOT_BOUND | P0 |
| Combined AGSi | SURMISE | — |

Packet: [`INDEPENDENT_KEEPER_PACKET.md`](INDEPENDENT_KEEPER_PACKET.md)

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --subject RG --adapter item --items science/agsi-eval/slice_b/wrap_items.json --log /tmp/rg.jsonl
```

Map: [`docs/science/CONSTELLATION_REMAINING_WORK.md`](../../docs/science/CONSTELLATION_REMAINING_WORK.md)
