# innovations-generator

Ranks at most three idea seeds from base ideas. Each seed has a rank, a title, a summary, a claim tier, the sources copied from the input, and a kill test.

The library calls `idea-recycling` to normalize and pair ideas, then writes the seeds. The `innovate` binary is `src/bin/innovate.rs`.

These crates are workspace members. They are outside the Tier-1 set in `TIER_MAP.md`, and `.github/workflows/core-tier1-ci.yml` does not test them. A green Tier-1 run does not cover this CLI.

Contact: info@Rathor.ai. License: AG-SML v1.1.

## Build and test

From the repo root:

```bash
cargo build -p idea-recycling -p innovations-generator
cargo test -p idea-recycling -p innovations-generator
cargo run -p innovations-generator --bin innovate -- input.json
```

Do not treat `cargo test --workspace` as a product-green result.

## `innovate` usage

```text
innovate [FILE]
innovate -              read stdin
innovate --help
```

No arguments also reads stdin. More than one argument exits 1. Invalid JSON exits 1 and prints `innovate:` plus the parse error on stderr.

Input is a JSON array of strings or objects, one object, or `{"base_ideas": [...]}`. Each object may set `text`, or `summary`, or `title` (the first non-empty of those three, in that order), plus `sources` and `evidence`.

Output is pretty JSON. The document is `{"seeds": [...]}`. Each seed has `rank`, `title`, `summary`, `claim_tier`, `sources`, and `kill_test`. The same input yields the same seeds. At most three seeds are written. Empty input writes an empty list.

`idea-recycling` drops blank text, merges duplicates, keeps at most 40 ideas, and pairs them. A pair's sort key is `shared_themes × 10 + source_count × 2 + combined_character_length / 40` (integer division). Seed selection keeps higher keys first and does not reuse a base idea to fill the cap. An unused idea is appended on its own, in recycled order.

Sources on a seed are the sources from its ideas, with blanks removed, duplicates removed, and the list sorted. The evidence note is not copied into the seed. It only affects `claim_tier`.

## Claim tiers

`claim_tier` records what the input attached. It is a label on the input, and the generator does not verify sources.

| Tier | When the code sets it |
|------|------------------------|
| `idea` | The seed has no source. An evidence note without a source stays `idea`. |
| `plausible` | The seed has at least one source, and at least one idea in the seed has no non-empty evidence note. |
| `evidenced` | The seed has at least one source, and every idea in the seed has a non-empty evidence note. |

The generator copies source strings and evidence notes. It does not open the sources, and it does not check that a note is true. `kill_test` is the falsification step: quote a passage that contradicts the claim, and discard the seed if you find one. When no source was attached, the kill test says to name one before treating the claim as checked. The CLI does not run that check.

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

Rank 3 is `evidenced` because the input attached `docs/REVENUE_SCALE.md` and the note `operator note`. That file was not opened, and the note was not checked.

## Creative-stuck idea recycling

When a real creative choice is not covered by a card or a ruling, run this CLI and post at most three seeds as a SUGGESTION on GitHub. The default is HOLD. A seed becomes a card only after the council votes and Sherif says OK. Merge routines do not call this CLI. A suggestion does not change Layer 0.

Pairing behavior is documented in [`../idea-recycling/README.md`](../idea-recycling/README.md). The 2026-09-07 human-side run record is [`../../docs/science/INNOVATIONS-GENERATOR-RUN-2026-09-07-TOLC-BASE-REALITY.md`](../../docs/science/INNOVATIONS-GENERATOR-RUN-2026-09-07-TOLC-BASE-REALITY.md).
