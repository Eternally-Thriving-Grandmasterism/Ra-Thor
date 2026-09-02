# QUICKSTART.md

**Ra-Thor workspace 14.15.6** · AG-SML v1.1 · [info@Rathor.ai](mailto:info@Rathor.ai)

Inspectable research software. Independent of xAI. Not certified. Not a legal product. Drafts need human review.

Full map: [`TIER_MAP.md`](TIER_MAP.md) · Front door: [`README.md`](README.md) · Longer tour: [`DEVELOPER-QUICKSTART.md`](DEVELOPER-QUICKSTART.md)

## 1. Clone and run the merge gate

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor

# Default members are TIER_MAP + mercy-security. This is the Core Tier-1 gate.
cargo test -p lattice-conductor-v14
cargo test -p mercy_tolc_operator_algebra
cargo test -p mercy-security
```

Do **not** `cargo test --workspace` and treat it as product-green. Research crates stay on disk; re-add a path in root `Cargo.toml` `members` to work on one. Conductor is **v14 only** (`lattice-conductor-v13` is deprecated).

## 2. ONE Organism (optional web demo)

```bash
cargo run -p ra-thor-one-organism --example one_organism_web_demo --features web-demo
```

## 3. Powrush (sibling repos, not this process)

| Surface | Repo |
|---------|------|
| Human-playable game | [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) |
| Browser client | [Powrush-MMO-Simulator](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO-Simulator) |
| Lattice telemetry / policy hints | this repo (`reality-thriving-transfer`, `crates/powrush`) |

Do not fold the player loop into Ra-Thor.

## 4. GitHub reads (agents and Grok)

Never recursive-root tree walks. Always `path_filter`. `per_page` ≤ 100. Prefer `get_file_contents_safe`. `get_tree_safe` walks to a subtree SHA and errors if GitHub truncates.

## Next

- [`TIER_MAP.md`](TIER_MAP.md) — what to build first
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PRs, PATSAGi, Layer 0
- [`PUBLIC_CLAIM.lock.md`](PUBLIC_CLAIM.lock.md) — public claim lock
- Contact **info@Rathor.ai** only

**Thunder locked in.** yoi ⚡
