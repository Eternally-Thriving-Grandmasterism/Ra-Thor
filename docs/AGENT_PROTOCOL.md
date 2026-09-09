# Agent protocol — Ra-Thor standing orders

Short orders for Grok, BabyBot, and Bot 1 on this repo.
This file does not replace the load-first path. It does not add a ninth gate.

**Contact:** info@Rathor.ai. Independent of xAI. Not certified. Not a legal product.
**License on main:** AG-SML v1.1. Workspace **14.15.6**. Do not bump identity to 15.x.

## Load first (already on main)

1. [`cursor-teams/AGENT_BOOT.md`](cursor-teams/AGENT_BOOT.md) — ADP bands, dual-gate merge, TOLC 8.
2. [`GROK_BOT_LATTICE_PATH.md`](GROK_BOT_LATTICE_PATH.md) — read one known path; never walk the repo root.
3. [`../TIER_MAP.md`](../TIER_MAP.md) — Core Tier-1 is the merge gate. Conductor is `lattice-conductor-v14` only.

If a rule here disagrees with those files, those files win. Do not fork a parallel protocol.

## Who does what

| Role | Repo | Does not |
|------|------|----------|
| BabyBot | This lattice (`Eternally-Thriving-Grandmasterism/Ra-Thor`) | WASD, Title Online, Market, Powrush-MMO client slices |
| Bot 1 | Powrush player loop (other repo) | Fold the game client into this monorepo |
| Any Grok | The crate named in the task | Recursive root walk, stub overwrite, secret dump |

If a task needs the game repo, stop and say so. Do not drive it from this seat.

## Before a change

1. Read current `main` first: `README.md`, `CHANGELOG.md`, root `Cargo.toml`, then the crate you will touch.
2. Search before you add. No duplicate crates. No second protocol file.
3. Name the files. Open that crate README and the target files only. Stop if unsure.
4. One slice = one PR. Smallest change that ships a real behavior.
5. Branch from current `main`. Do not paste-overwrite `main`.

## Merge

- Propose. Merge only when Core / Tier-1 is green (`.github/workflows/core-tier1-ci.yml`, `TIER_MAP.md` `-p` list).
- Do not run `cargo test --workspace` and call it product-green.
- Do not invent eval scores. If evidence is missing, list the gap.

## HOLD (steward only)

- Public bind, including listen on `0.0.0.0`.
- Lethal-as-default.
- Folding the Powrush client into this repo.
- Contact-email as a required CI gate when it is satellite noise.

Cosmic Loop, TOLC 8, and mercy gates already on a path you touch stay enforced. Do not invent a 9th gate that breaks 8.

## Never

- Overwrite a file with a stub or a placeholder. A one-line stub is a fail beat.
- Always-allow a GitHub write that publishes Pages, binds `0.0.0.0`, or dumps secrets.
- Drive Powrush WASD, Title Online, or Market from this repo.
- Treat `legal-lattice`, `mercy_predictive_policing`, or `mercy_shield_law_enforcement` as shipped products.

## Next slice

R4 is standing-doc honesty (this PR).
R5 is a fresh constellation pointer from the current tip, not stale #445.
R6 is one fail-closed Cosmic Loop test in one named Tier-1 crate, Steward-named later.

Do not start R5 or R6 in this PR.
