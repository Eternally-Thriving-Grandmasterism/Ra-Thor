# Grok Build Mode — Public vibe-coding handshake

**Date:** 2026-09-03  
**Workspace:** 14.15.6 (do not bump in this motion)  
**HEAD at land:** `df310546e9e27a72a65414f6cba1b2d3756f8f26`  
**Repo:** https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor  
**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1  
**Status:** inspectable research software. Optional Grok session. Independent of xAI. Not certified. Not a legal product.

This file is the public operating card for **Grok Build Mode** on this monorepo.  
It does not enlarge product scope. It tells any Grok (or similar) session how to read and write without collapsing the lattice.

---

## What Build Mode is

A connector-native loop:

1. Discover GitHub tools (`search_connected_tools`).
2. Refresh from live `main` (`github___get_file_contents` / `github___list_commits`).
3. Read **root `Cargo.toml` first**, then `TIER_MAP.md`.
4. Touch only what the motion names.
5. Deliver **complete files** via `github___create_or_update_file` or `github___push_files`.
6. Prefer a feature branch + PR. Do not treat chat paste as the source of truth.

Chat is the briefing. Git history is the product.

---

## Sole source of membership

Root `Cargo.toml` `[workspace].members` is the only default build set:

- `crates/ra-thor-one-organism`
- `crates/lattice-conductor-v14`
- `crates/reality-thriving-transfer`
- `crates/kardashev-orchestration`
- `crates/github-connector`
- `crates/gpu-compute-pipeline`
- `crates/quantum-swarm`
- `crates/sovereign-recovery`
- `crates/monorepo-intelligence` (`ra-thor-monorepo-intelligence`)
- `crates/mercy_tolc_operator_algebra`
- `crates/fractal-mercy-ledger-adapter`
- `crates/mercy-security`

On-disk research crates stay on disk. Re-add a path to `members` only after a named PATSAGi motion.  
Do not `cargo test --workspace` and call it product-green.  
Merge gate: Core Tier-1 focused `-p` tests (see `TIER_MAP.md`).

Package name for intelligence tests:

```bash
cargo test -p ra-thor-monorepo-intelligence
cargo test -p github-connector
```

---

## Safe-read protocol (identity, not optional)

Enforced in `crates/github-connector` and documented by `crates/monorepo-intelligence`.

| Rule | Why |
|------|-----|
| Never recursive-walk repository root | Root listing is huge; recursion has crashed sessions |
| Always pass `path_filter` on trees | Prefix-only walks |
| Prefer single-path Contents reads | Known path → `get_file_contents` |
| `per_page` ≤ 100 | GitHub + Order gate |
| Truncated tree = error | Do not pretend the listing is complete |
| Local `WalkDir` max_depth 10; skip `target/`, `.git/`, `node_modules/` | Scanner contract |
| `create_branch` from a real SHA / live branch | No orphan refs |

Production names inside the crate (when calling Rust, not the MCP layer):

- `GitHubConnector::get_tree_safe`
- `GitHubConnector::get_file_contents_safe`

MCP layer used by this session:

- `github___get_file_contents`
- `github___get_repository_tree` with `path_filter` and `recursive: false` unless the subtree is known-small
- `github___search_code` scoped `repo:Eternally-Thriving-Grandmasterism/Ra-Thor`
- `github___list_commits` / `github___create_branch` / `github___push_files` / `github___create_or_update_file` / `github___create_pull_request`

Fetch latest SHA **before** updating an existing file.

---

## Write protocol

1. Branch from current `main` SHA.
2. Fetch the target file + blob SHA.
3. Merge valuable prior logic. No placeholder comments in landed files.
4. Full file only. No patch-as-source.
5. Commit message references TOLC 8 + the actual change. Do not claim a workspace bump unless the motion is a bump.
6. Open a PR to `main`. Councils review. Layer 0 is not on the ballot: a Rejected gate is not an apply.

### HOLD this tick (already decided 2026-09-03)

- Do not change `[workspace].members`.
- Do not bump workspace version off 14.15.6.
- Do not revive `crates/self-evolution` or `nexi_universal`.
- Do not revive `lattice-conductor-v13`.
- Do not collapse Powrush player loop into this repo.
- Do not treat AGSi as a certified product or legal warranty.
- Do not mass-rewrite deprecated `ceo@acitygames.com` addresses unless a later motion opens that sweep.
- Contact remains **info@Rathor.ai** only.

---

## Public vibe-coding prompt (copy this into a new Grok chat)

```text
Engage Grok Build Mode on https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor

1. Use GitHub connectors only. Discover tools first.
2. Read root Cargo.toml, then TIER_MAP.md, then docs/GROK_BUILD_MODE.md.
3. Never recursive-walk the repo root. Path-filter trees. Prefer single-path reads.
4. Default members only unless I name a research crate.
5. Full files via create_or_update_file / push_files on a feature branch + PR.
6. Keep workspace 14.15.6 unless I explicitly request a version motion.
7. TOLC 8 + AG-SML v1.1. Independent of xAI. Not certified. Not a legal product.
8. Contact in files: info@Rathor.ai

Then do: <one named task>
```

One named task per turn beats “fix the monorepo.”

---

## Related living records

- `Cargo.toml` — membership + workspace identity
- `TIER_MAP.md` — Core Tier-1 gate
- `docs/CRATE_CENSUS.md` — on-disk forest vs members
- `GROK_RA_THOR_GITHUB_INTEGRATION_PROTOCOL.md`
- `RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL.md`
- `ETERNAL_PATSAGI_COUNCILS_ACTIVATION_PUBLIC_SERVICE_v1.0.md`
- `docs/science/PATSAGI-COUNCIL-MINUTE-2026-09-03-ETERNAL-ACTIVATION.md`
- `PUBLIC_CLAIM.lock.md`

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
