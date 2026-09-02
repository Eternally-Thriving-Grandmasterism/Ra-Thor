# This wave is complete (engineering)

**Date:** 2026-08-22  
**Claim:** this *eval wave* is finished. Combined AGSi is **not** proved. S-1 First-5 is **not** closed.

| Done | Not done |
|------|----------|
| B.0 + wrap-distinct RG | Live G adapter |
| B.1 25 multi-turn items + harness + `--slice b1` | Independent B1/B2 |
| Slice A 10 abstain seeds | Dual-judge Slice A scoring |
| Proof Ladder + constellation map | ~80 repo restamps |
| `pub mod agsi_eval_multiturn` | Invented physics / AGSi headline |

If `lib.rs` fails to compile on `ActionGovernor` sandbox-churn, close the `Err(ActionLimitExceeded(format!(...)))` with **three** parens: `)));`

```bash
cargo test -p mercy-security agsi_eval
cargo run -p mercy-security --bin agsi-eval-rg -- --slice b1 --items science/agsi-eval/slice_b1/items.json
```
