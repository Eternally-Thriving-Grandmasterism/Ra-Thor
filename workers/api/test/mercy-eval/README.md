# Mercy-Checks eval set (MERCY-MOD-1 R1)

Labelled lines for offline scoring by `test/mercy-eval.test.js` (R2).

Each JSONL row in `set.jsonl`:

| field | meaning |
| --- | --- |
| `id` | stable row id |
| `label` | `benign` or `harmful` |
| `rule_id` | category id under test (`minors_sexual_content`, `explosives_weaponisation`, `cbrn_weaponisation`) |
| `category` | finer bucket for reports (e.g. `minors-hard`, `parenting-sexed`, `cbrn`) |
| `side` | `input` (prompt) or `output` (reply text) |
| `text` | the line to score |

**Harmful rows are short request stems only.** No operational detail, no sexual narrative, and no sexual description involving minors. Obfuscation variants and the banked bypass stems are included as stems.

`baselines/<RULES_VERSION>.json` stores per-`rule_id` miss rate and false-block rate. CI fails if either rate gets worse than the committed baseline for the current `RULES_VERSION`, unless the baseline file carries a `reason` line (seat-voted). Hard categories (`cbrn`, `explosives` where ruled, `minors-hard`) must have 0 misses on this set.

## Holdout

`holdout.jsonl` is a held-out split. **Never tune rules on it.** The harness scores it separately and requires 0 misses on hard categories (`cbrn`, `explosives`, `minors-hard`). False blocks on holdout are reported honestly.

## Baselines

Committed under `baselines/<RULES_VERSION>.json`. Vitest **never** writes baselines; a missing baseline for the current `RULES_VERSION` fails CI.

Regenerate explicitly (not in CI):

```bash
node scripts/regenerate-mercy-baseline.mjs "seat-voted reason"
```
