# idea-recycling

Normalizes base ideas, extracts themes, and pairs them. Pairings are text operations. They are seeds for a later check. This crate has no binary.

`innovations-generator` calls this library and writes at most three ranked seeds. The command is:

```bash
cargo run -p innovations-generator --bin innovate -- input.json
```

Usage, claim tiers, and a captured sample live in [`../innovations-generator/README.md`](../innovations-generator/README.md). The sample below is that same run.

Contact: info@Rathor.ai. License: AG-SML v1.2.

## What `recycle` does

- Trims text and collapses whitespace. Blank text is dropped.
- Duplicates match on the first 12 alphanumeric tokens, lowercased. Their sources are merged. The first non-empty evidence note in input order is kept.
- Themes come from the idea's own text. A token is kept when it is in a short lexicon, or when `token.len()` is at least 5. That length is UTF-8 bytes, not letters. The token must contain a letter. Stopwords are dropped. At most 12 themes are kept.
- After that, at most 40 ideas are kept. Order is highest signal first, then text. Signal is `theme_count × 3 + character_length / 40` (integer division).

## What `cross_pollinate` does

Two or more ideas become every unordered pair. One idea becomes a one-idea pairing. Pair order is stable. The sort key is `shared_themes × 10 + source_count × 2 + combined_character_length / 40` (integer division), then the idea text and sources. That key is not a measurement of truth.

`innovations-generator` then keeps higher-sorting pairs first, does not reuse a base idea to fill a cap of three, and appends an unused idea on its own.

## Build and test

From the repo root:

```bash
cargo build -p idea-recycling -p innovations-generator
cargo test -p idea-recycling -p innovations-generator
```

Both crates are workspace members. They are outside the Tier-1 set in `TIER_MAP.md`. Do not treat `cargo test --workspace` as a product-green result.

## Claim tiers

This crate does not assign `claim_tier`. The generator does, from what the input attached:

| Tier | When the generator sets it |
|------|----------------------------|
| `idea` | The seed has no source. |
| `plausible` | The seed has at least one source, and at least one idea in the seed has no non-empty evidence note. |
| `evidenced` | The seed has at least one source, and every idea in the seed has a non-empty evidence note. |

The generator does not verify sources. It does not open them, and it does not check that an evidence note is true. `kill_test` is the falsification step: quote a passage that contradicts the claim, and discard the seed if you find one. The CLI does not run that check.

## Example

Captured from `cargo run -p innovations-generator --bin innovate -- input.json` on commit `a3c15e3f` (2026-10-16).

`input.json`:

```json
{
  "base_ideas": [
    {"text": "Displacement is a public-service concern", "sources": ["notes/layer-a.md"]},
    {"text": "Soft feedback events are sealed telemetry", "sources": ["notes/layer-b.md"]},
    {"text": "Abundance claims need a cash path", "sources": ["docs/REVENUE_SCALE.md"], "evidence": "operator note"},
    {"text": "Proof ladder is a queue for surmise", "sources": []},
    {"text": "A research label stays a surmise until a named check", "sources": ["docs/science/minute.md"]}
  ]
}
```

Stdout:

```json
{
  "seeds": [
    {
      "rank": 1,
      "title": "surmise: A research label stays a surmise until a name... × Proof ladder is a queue for surmise",
      "summary": "Seed from 2 base ideas: A research label stays a surmise until a named check | Proof ladder is a queue for surmise. Shared themes: surmise.",
      "claim_tier": "plausible",
      "sources": [
        "docs/science/minute.md"
      ],
      "kill_test": "Falsify \"A research label stays a surmise until a named check\" and \"Proof ladder is a queue for surmise\". Look first in \"docs/science/minute.md\". Quote a passage that contradicts the claim. If you find that passage, discard this seed."
    },
    {
      "rank": 2,
      "title": "Displacement is a public-service concern × Soft feedback events are sealed telemetry",
      "summary": "Seed from 2 base ideas: Displacement is a public-service concern | Soft feedback events are sealed telemetry. Shared themes: none.",
      "claim_tier": "plausible",
      "sources": [
        "notes/layer-a.md",
        "notes/layer-b.md"
      ],
      "kill_test": "Falsify \"Displacement is a public-service concern\" and \"Soft feedback events are sealed telemetry\". Look first in \"notes/layer-a.md\". Quote a passage that contradicts the claim. If you find that passage, discard this seed."
    },
    {
      "rank": 3,
      "title": "abundance: Abundance claims need a cash path",
      "summary": "Seed from 1 base idea: Abundance claims need a cash path. Themes: abundance, claims.",
      "claim_tier": "evidenced",
      "sources": [
        "docs/REVENUE_SCALE.md"
      ],
      "kill_test": "Falsify \"Abundance claims need a cash path\". Look first in \"docs/REVENUE_SCALE.md\". Quote a passage that contradicts the claim. If you find that passage, discard this seed."
    }
  ]
}
```

Rank 3 is `evidenced` because the input attached a source and a note. Neither was checked.
