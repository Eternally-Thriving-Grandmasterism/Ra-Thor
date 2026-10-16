# STUB-AUDIT-1

**Date:** 2026-10-16
**Tip audited:** `5c698566ce4f99b7ce8abc0df9ac34217d1164bd`
**Workspace identity:** 14.15.6
**Contact:** info@Rathor.ai
**Status:** inspectable inventory. Not a certification. Not a build of the research forest.

This file is the canonical list of Rust manifests that do not build, or that build only as stubs. After it merges, each family below is one make-real card. This audit adds no crate, no member, and no workflow.

Conductor in the living set is v14 only (`crates/lattice-conductor-v14`). Do not revive `lattice-conductor-v13`.

Public TOLC 8 names, when a later card talks about mercy checks: Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony. First mention of the gates in code: TOLC 8 mercy checks (Mercy Gates in code).

---

## Already fixed (DONE)

These were make-real cards. `cargo check` on 2026-10-16 finished for each. They are standalone workspaces. They are not root default members.

| Path | PR | What is true now |
|------|----|------------------|
| `crates/fenca` | #661 | Audit can return `passed=false`. Online `cargo check --manifest-path crates/fenca/Cargo.toml` finished. |
| valence enforcer (fenca audit bin + `scripts/check-valence-enforcer.sh`, not a separate crate) | #663 | Fail-closed on a real fenca audit. |
| `crates/council` | #665 | Standalone `[workspace]`. Offline `cargo check` finished in 0.82s. A proposal can fail. An admission block cannot be voted into an apply. |
| `crates/powrush` | #668 | Standalone `[workspace]`. Online `cargo check` finished. Policy hints can be rejected. |
| `crates/sovereign-shard-genesis` | #669 | Standalone `[workspace]`. Online `cargo check` finished. SHA-256 of explicit inputs. The v13 draft file is not a module. |
| LEAF-BUILDS-1: `crates/legal-research-source-locked`, `crates/hyperbolic-neural-network`, `crates/hyperbolic-tiling-consciousness`, `crates/moebius-transformations`, `crates/philotic-web-fusion`, `crates/rathor-sovereign-reasoning-engine`, `crates/shared-valence-field` | #671 | Standalone `[workspace]` resolver 2 and a committed `Cargo.lock` each. Not root members. Locked `cargo build`, `cargo test`, and `cargo clippy -- -D warnings` finished. Threshold literals stay source literals. |

Two tidy leftovers on the shard crate are banked below. They did not stop `cargo check`.

---

## How this was checked

Tool versions: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, Cargo 1.99.1.

1. Every `Cargo.toml` outside `target/` was enumerated. Count: **371**.
2. Root default members (15) were not re-checked. Core Tier-1 CI owns them. This seat did not run `cargo test --workspace`.
3. The four DONE standalones were checked with `cargo check --manifest-path` (online, except `council`, which also finished offline).
4. The other **351** manifests were checked with `cargo check --manifest-path <path> --offline`. Every one failed before producing a library. First errors are manifest errors. Type errors behind those manifests were not reached, except for the leaf probe below.
5. Leaf probe, not committed: 25 non-member crates whose manifests have no `workspace = true` and no `path` dependency were copied to `/tmp` and given an empty `[workspace]` table (the #665 / #668 / #669 pattern). `cargo check` was then run in that copy. `web-forge`, `llama-cpp-gguf`, and `wasm-ml-candle` were not probed (heavier dependency trees).
6. Stub search under `crates/**/*.rs`: no `unimplemented!` and no `todo!` macros. Comment `TODO`s, the word `stub`, functions that return `true` or `Ok(())`, and numeric literals such as `0.999999` were read in the files cited below. Those literals are source constants. They are not measurements.

A non-member inside this repo fails `cargo check --manifest-path` in one of two ways:

- `current package believes it's in a workspace when it's not` (70 crates under `crates/`, plus benches, fuzz, src-tauri, and some archive manifests). Cargo's own hint is the standalone fix: add an empty `[workspace]` table, or add the path to root `members`. Adding it to root `members` makes Cargo load that manifest on every `cargo test -p`. That needs Core. The standalone table does not.
- `dependency.<key> was not found in workspace.dependencies` when a `workspace = true` key is missing. Root `[workspace.dependencies]` has no `powrush`, no `patsagi-councils`, and no `ra-thor-kernel`.

`anyhow` is also absent from root `[workspace.dependencies]`. Two crates fail on it first: `crates/lattice-conductor` and `crates/spacetime-reducer-bridge`.

---

## Counts at this tip

| Set | Count |
|-----|------:|
| `Cargo.toml` files audited | 371 |
| under `crates/` | 268 |
| under `docs/archive/` | 98 |
| other (`Cargo.toml`, `benches/`, `examples/`, `fuzz/`, `src-tauri/`) | 5 |
| root default members | 15 |
| standalone workspaces that check | 4 |
| non-members whose in-repo `cargo check` failed | 351 |
| `crates/` manifests with no `*.rs` sources | 95 |
| `crates/` top-level directories with no `Cargo.toml` | 31 |
| manifests with exact `powrush = { workspace = true }` | 165 |
| of those, first cargo error is the missing `powrush` key | 8 |

`docs/CRATE_CENSUS.md` and `docs/FOREST_TRIAGE.md` still say 12 default members and 267 manifests under `crates/`. This tip has 15 members (the old 12, plus `crates/mercy`, `crates/idea-recycling`, and `crates/innovations-generator`) and 268 manifests under `crates/` (`shard-composer` included). This audit does not rewrite those files.

---

## Family order

Dependencies first. Then the smallest crates whose source already checks once the manifest is standalone. Fictional and empty families last. Names are the future cards. This file does not open those cards.

| Order | Card | Crates | Why it is here | Size | Core? |
|------:|------|--------|----------------|------|-------|
| 1 | `LEAF-BUILDS-1` (DONE, #671) | 7 leaves that compiled in the probe | No path deps. Standalone `[workspace]` is the whole manifest fix. Still read the constant floors before calling them real. | S | No, if they stay out of root `members`. |
| 2 | `ONE-ERROR-LEAVES-1` | 3 | Same manifest fix, then one Rust error each. | S | No, if standalone. |
| 3 | `SHARED-VALENCE-DEPENDENTS-1` | 5 path dependents of `shared-valence-field` | Depends on card 1. | S | No, if standalone. |
| 4 | `MIAL-THRESHOLD-BUILDS-1` | `mial`, `mercy-threshold-wasm` | `powrush-mmo-simulator` path-deps both. | M | No, if standalone. |
| 5 | `RA-THOR-CORE-BUILDS-1` | `crates/ra-thor-core` | `patsagi-councils` path-deps it. | M (988 lines, 9 files) | No, if standalone. |
| 6 | `QSO-BUILDS-1` | `crates/quantum-swarm-orchestrator` | `patsagi-councils` path-deps it. It path-deps missing `self-evolution`. | L (9,872 lines) | No, if standalone. Do not add a `self-evolution` crate. |
| 7 | `PATSAGI-COUNCILS-BUILDS-1` | `crates/patsagi-councils` | After 5 and 6, and after `powrush` (#668). Living small council is already `crates/council`. | L (6,115 lines, 26 files) | No, if standalone. Yes if added to root `members`. |
| 8 | `POWRUSH-MMO-SIM-BUILDS-1` | `crates/powrush-mmo-simulator` | After 4 and #668. Not the Powrush-MMO repo and not the browser client. | M (1,039 lines) | No, if standalone. |
| 9 | `SHARD-COMPOSER-PARSE-1` | `crates/shard-composer` | TOML does not parse. One-line array break. | S | No. |
| 10 | `KERNEL-MANIFEST-1` | `crates/ra-thor-kernel`, `crates/kernel` | Unstable `rustflags`, missing path deps. Separate from the undefined workspace key `ra-thor-kernel`. | S–M | No, if not a default member. |
| 11 | `REASONING-COUNCIL-WIRES-1` | council-named crates that import crates they do not depend on | After card 1 (`rathor-sovereign-reasoning-engine`, `philotic-web-fusion`, `moebius-transformations`). Label fleet and post-quantum strings as design-only inside the same card. | S each | No, if standalone. |
| 12 | `CONDUCTOR-LEGACY-RETIRE-1` | unversioned conductor, v13 and its examples, `tolc8-genesis-gate` | Deprecated or fictional. Do not make them the living conductor. | L on disk | No. A v14 edit would need Core. This card should not edit v14. |
| 13 | `NEXI-PATH-RETIRE-1` | crates with `nexi = { path = "../" }` | That path is not a package. | many manifests | No. |
| 14 | `WORKSPACE-KEY-SHELLS-1` | 165 `powrush` workspace deps, 95 sourceless manifests, string ZK and propulsion stubs | Shared missing workspace keys. Most have nothing to compile, or return constants. Design-only or leave on disk. | manifests, plus a few short files | No. Do not add them to `members`. |
| 15 | `ARCHIVE-LINEAGE-RETIRE-1` | 98 manifests under `docs/archive/` | Copies. 76 fail loading `nexi`. | archive | No. |

Banked tidy, not make-real of a crate: `POWRUSH-TIDY-1`, `SHARD-GENESIS-TIDY-1`, `STANDALONE-CI`. See the last section.

---

## 1. LEAF-BUILDS-1 (DONE, #671)

**Status:** DONE in #671. Each crate below is its own workspace (`resolver = "2"`) with a committed `Cargo.lock`. They are not root members. Locked `cargo build`, `cargo test`, and `cargo clippy -- -D warnings` finished. Threshold literals stay in the source and are labelled as source literals.

In-repo first error for each, before that PR: `current package believes it's in a workspace when it's not`.

Probe (empty `[workspace]` in `/tmp`): `cargo check` exited 0 and compiled a library.

| Path | Probe | What the source actually is |
|------|-------|------------------------------|
| `crates/legal-research-source-locked` | compiled | The crate docs already say research sketch, draft, not a legal product (`src/lib.rs` lines 1–11). Smallest honest leaf. |
| `crates/hyperbolic-neural-network` | compiled | `tanh` projection. `forward` rejects `valence < 0.9999999` (`src/lib.rs` lines 18–30). The literal is not a measured gate. |
| `crates/hyperbolic-tiling-consciousness` | compiled | Poincaré disk points (`src/lib.rs` lines 8–19). Header names a council. The math that compiled is the point type. |
| `crates/moebius-transformations` | compiled | Möbius matrix over `nalgebra::Complex` (`src/lib.rs` lines 5–24). |
| `crates/philotic-web-fusion` | compiled | `fuse_bond` returns `Err` when `valence < 0.9999999`. Empty web returns that same literal (`src/lib.rs` lines 23–35). Constant, not a result. |
| `crates/rathor-sovereign-reasoning-engine` | compiled | `MERCY_THRESHOLD` is the literal `0.9999999`. `new` sets `valence: 0.99999997` (`src/lib.rs` lines 12–37). The file header uses superintelligence language. The type is a struct with those literals. This audit did not run it. |
| `crates/shared-valence-field` | compiled | `new` sets `collective_valence: 0.999999`. `emit` then floors with `.max(0.999999)` (`src/lib.rs` lines 46–54). Constant floor, not a measurement. |

`crates/mercy-gel-pterosaur-propulsion-council` also compiled. It is not in this card. `valence_threshold: 0.9999999`, and `activate_mercygel_production` returns a string (`src/lib.rs` lines 17–25). Pterosaur-wing propulsion and a post-quantum fleet are design-only. Card 14.

`crates/geometric-intelligence` is not in this card. In-repo error is the membership error. The `/tmp` probe exited 0 in 0.00s because there is no `src/lib.rs` (only `tests/`). `cargo check` did not compile a library. `cargo test` was not run.

**Fix:** empty `[workspace]`, pin the crates.io deps already in the manifest, same pattern as #665. Do not copy the `0.999999` literals forward as if they were eval scores. Do not add these paths to root `members`.

**Runtime / engine / v14:** none of these are default members. No v14 edit.

**Size:** S. Each is one small library.

---

## 2. ONE-ERROR-LEAVES-1

In-repo first error: not a workspace member. Probe then showed one Rust error.

| Path | Next error (probe) |
|------|--------------------|
| `crates/swarm_intelligence` | `src/lib.rs:83`: no method `replicate` on `SwarmAgent`. The call is commented as a placeholder. |
| `crates/patsagi-quantum-valence` | `src/lib.rs:29`: `sqrt` on an ambiguous integer in `valence_qaoa`. The function returns that vector unchanged (lines 27–31). `valence_vqe` returns `hamiltonian.trace() * valence_weight` (lines 35–37). Names say QAOA and VQE. The bodies are not those algorithms. |
| `crates/hyperon-metta-pln` | `src/lib.rs:9`: doc comment is not attached to an item. |

**Fix:** Use a standalone `[workspace]`, then fix the one error. For `patsagi-quantum-valence`, label the QAOA/VQE names as design-only, or reword the claim in the function docs (with a council note). Do not fill them with a fake quantum stack.

**Runtime / v14:** no.

**Size:** S (`swarm_intelligence` 93 lines, the other two are single files).

---

## 3. SHARED-VALENCE-DEPENDENTS-1

Depends on `crates/shared-valence-field` (card 1). In-repo first error for each: not a workspace member. Compile after the manifest fix was not run. Each path-deps `../shared-valence-field`.

- `crates/abundance-breath`
- `crates/epiphany-bridge`
- `crates/resonance-challenge`
- `crates/soft-sovereign-agency`
- `crates/symbiotic-membrane`

`crates/shared-valence-field/src/dual_repo_healh.rs` line 23 is a `TODO` to bind a dual-repo health endpoint. The filename is spelled `healh`. Unknown whether these five crates call it.

**Fix:** standalone workspace each, path-dep the standalone field crate. Keep the constant floor labelled as a constant.

**Runtime / v14:** no.

**Size:** S. Not measured past the manifests.

---

## 4. MIAL-THRESHOLD-BUILDS-1

`crates/powrush-mmo-simulator` path-deps both.

| Path | In-repo first error | Notes |
|------|---------------------|-------|
| `crates/mial` | not a workspace member | Path-deps `../mercy_gating_runtime`. 1,416 lines, 16 files. |
| `crates/mercy-threshold-wasm` | not a workspace member | Path-deps `../mial`. 232 lines. Feature `native` is what the simulator asks for. Unknown whether that feature is more than a name. |

`crates/mercy_gating_runtime` is the next dependency. In-repo: not a member. Probe (it has no path deps): 23 errors, including `src/lib.rs:10` file not found for module `error`, and `serde` used but not linked. 754 lines. It is part of this card, ahead of `mial`.

**Fix:** Use standalone workspaces from the bottom up. For the gating runtime, repair the missing modules or retire them (with a council note). Then build `mial`, then `mercy-threshold-wasm`. Do not pull them into root `members`.

**Runtime / v14:** no, unless a later card wires the wasm crate into the site. That would be a different card.

**Size:** M.

---

## 5. RA-THOR-CORE-BUILDS-1

`crates/ra-thor-core`. In-repo first error: not a workspace member. 988 lines, 9 files. Compile past the manifest was not run.

`crates/patsagi-councils/Cargo.toml` line 12: `ra-thor-core = { path = "../ra-thor-core" }`.

**Fix:** standalone `[workspace]` if its own deps allow it. Read the manifest before adding crates.io crates. Do not invent modules to satisfy `patsagi-councils`.

**Runtime / v14:** no.

**Size:** M.

---

## 6. QSO-BUILDS-1

`crates/quantum-swarm-orchestrator`. In-repo first error: not a workspace member. 9,872 lines, 65 files.

Path dep `self-evolution` points at `../../self-evolution`. `crates/self-evolution` has **no** `Cargo.toml`. `docs/CRATE_CENSUS.md` already says it is not a crate and not a product. Do not add a manifest.

`src/orchestrator.rs` lines 58–60 are `TODO` comments for polyhedral mode, a Riemannian manifold, and resonance handlers. Not implemented.

`crates/quantum` (6,906 lines of `*.rs` in the earlier forest count) is a directory with **no** `Cargo.toml`. It is not this crate. Do not promote it.

**Fix:** standalone workspace only for the orchestrator package. Drop or feature-gate the `self-evolution` path. Leave the TODO methods unimplemented until a designed card, and do not stub them with a constant score.

**Runtime / v14:** the default member is `crates/quantum-swarm`, a different package. This card must not replace it. No v14 edit.

**Size:** L.

---

## 7. PATSAGI-COUNCILS-BUILDS-1

`crates/patsagi-councils`.

**In-repo first error:** not a workspace member. Package fields use `version.workspace = true` and the rest of the workspace inheritance block (`Cargo.toml` lines 3–7). Dependencies include `chrono.workspace = true`, `serde`, `serde_json`, `thiserror`, `uuid` (`Cargo.toml` lines 22–26).

**Next blockers, read from the manifest, not compiled:**

- path `../ra-thor-core` (card 5)
- path `../powrush` (DONE, #668, but it is its own workspace; a path dep can work)
- path `../mercy` (default member `mercy`)
- path `../quantum-swarm-orchestrator` (card 6)
- `lean-sys = "0.1"` (`Cargo.toml` line 29). The comment says an FFI bridge is active. That was not demonstrated.
- optional paths: `real-estate-lattice`, `ra-thor-post-quantum-sig`, `interstellar-operations`

6,115 lines, 26 files, plus examples and `src/bin/council_simulator.rs`.

`src/lib.rs` lines 1–8 describe 16 councils and a living tick. That is the file header, not a test result.

The living council crate that already builds is `crates/council` (#665). This package is the older monolith. A make-real card should not pretend they are the same crate.

**Fix:** standalone `[workspace]`, pin serde/chrono/uuid, path-dep only crates that already check (`mercy` is a default member and can be a path dep; `powrush` can be a path dep). Cut `lean-sys` until a crate of that name is shown to resolve. Cut optional features that point at non-building crates. Keep reject / refine / human-override visible. A council vote must not flip Reject to Apply (already true in `crates/council`).

**Fictional parts:** post-quantum demo example name, world-governance engine, self-evolving mercy core. Label those modules design-only if they do not compile after the cut. Do not replace them with constant scores.

**Runtime / v14:** not a default member. Adding it to root `members` needs Core, because Cargo loads every member manifest.

**Size:** L.

---

## 8. POWRUSH-MMO-SIM-BUILDS-1

`crates/powrush-mmo-simulator`.

**In-repo first error:** not a workspace member (`ra-thor-mercy`, `ra-thor-quantum-swarm-orchestrator`, `ra-thor-evolution`, `ra-thor-plasticity-engine-v2`, `serde`, `tokio`, and others are `workspace = true`). `ra-thor-mercy` **is** a root workspace key (alias to `crates/mercy`). The membership error fires before a missing-key error because those keys exist.

**Path deps:** `../powrush` (#668), `../mial`, `../mercy-threshold-wasm` (card 4).

1,039 lines. Examples under `examples/` are part of this package, not separate crates.

`Cargo.toml` lines 6–8 call this a Resource-Based Economy simulation. Resource-Based Economy is a design thesis, not a present fact. README already says the browser client is a different repository, and Powrush-MMO is a separate human-game repo. This crate is neither.

**Fix:** standalone workspace, path-dep `powrush`, `mial`, and `mercy-threshold-wasm` only after card 4. Drop workspace aliases that point at non-members (`ra-thor-evolution`, `ra-thor-plasticity-engine-v2`) unless those cards exist. Do not start Powrush-MMO work from this repo.

**Runtime / v14:** no.

**Size:** M.

---

## 9. SHARD-COMPOSER-PARSE-1

`crates/shard-composer`.

**First error:** TOML parse. `Cargo.toml` lines 40–43. The `focused-geometry` array is not closed, so line 43 is `unexpected '=' in array`.

```
focused-geometry = [
    "geometric-intelligence/focused-geometry",

real-estate = ["focused-real-estate"]
```

The manifest also uses `workspace = true` for package fields and for `ra-thor-mercy`, `serde`, `serde_json`. After the bracket is fixed, the next error is the membership / missing-key class. Not compiled.

**Fix:** close the array. Then decide whether the crate should be a standalone workspace. `geometric-intelligence` has no library (see card 1). Do not enable the `full` feature against crates that do not build.

**Runtime / v14:** no.

**Size:** S for the parse. Unknown after that.

---

## 10. KERNEL-MANIFEST-1

Two different directories.

| Path | First error |
|------|-------------|
| `crates/ra-thor-kernel` | `profile-rustflags` is required and is not stable in Cargo 1.99.1. `[profile.release] rustflags` is set at `Cargo.toml` lines 18–24. Path deps `../dilithium_sign` and `../falcon_sign` do not exist (no such directories). |
| `crates/kernel` | not a workspace member. 499 lines, 10 files. Package name is `kernel`. Root workspace.dependencies does **not** define a key `ra-thor-kernel`. |

`crates/council` comments (`Cargo.toml` lines 16–19) already record that the old `ra-thor-kernel` path was `../kernel` and that council does not depend on it.

**Fix:** Retire the unstable `rustflags` profile from `ra-thor-kernel` (with a council note), or move it to `.cargo/config.toml` only if a wasm card asks for it. Remove path deps on missing `dilithium_sign` and `falcon_sign`. Do not invent post-quantum implementations. `crates/kernel` can be a standalone workspace in its own card if something still calls it. Do not add a root key `ra-thor-kernel` just to satisfy the 167 manifests that name it (card 14).

**Runtime / v14:** no.

**Size:** S for `ra-thor-kernel` (manifest only plus whatever sources exist). M for `crates/kernel`.

---

## 11. REASONING-COUNCIL-WIRES-1

These import `rathor_sovereign_reasoning_engine` (and sometimes `philotic_web_fusion` or `moebius_transformations`) without declaring the dependency. In-repo first error is still "not a workspace member", because that fires before rustc. The probe (empty `[workspace]`, no new deps) showed the missing crates.

| Path | Probe error |
|------|-------------|
| `crates/eternal-nexus-revelation-council` | unresolved `rathor_sovereign_reasoning_engine` (`src/lib.rs:5`) |
| `crates/universal-sovereign-abundance-lattice-council` | same, `src/lib.rs:6` |
| `crates/post-quantum-biomimetic-grokarena-encryption-council` | same, `src/lib.rs:5` |
| `crates/mercy-quanta-sentinel-council` | unresolved reasoning engine and `philotic_web_fusion` (`src/lib.rs:6–7`) |
| `crates/quantum-gravity-harmony-council` | same pair (`src/lib.rs:6–7`) |
| `crates/quantum-propulsion-sovereignty-council` | unresolved reasoning engine and `moebius_transformations` (`src/lib.rs:5–6`) |

These path-deps are already declared, so the probe did not cover them. In-repo first error: not a member. Compile not reached.

- `crates/dynamic-persona-router-council`
- `crates/eternal-sovereign-divine-spark-council`
- `crates/eternal-sovereign-infinite-horizon-council`
- `crates/eternal-sovereign-lattice-completion-council`
- `crates/eternal-sovereign-mercy-lattice-unification-council`
- `crates/eternal-sovereign-quantum-consciousness-expansion-council`
- `crates/eternal-sovereign-spark-council`
- `crates/infinite-horizon-exploration-council`
- `crates/sovereign-decentralized-propulsion-fleet-council`

`crates/biomimetic-sovereign-harmony-council` has a `Cargo.toml` and **no** Rust sources (`no targets specified`).

**Fictional, label design-only rather than implement:** post-quantum biomimetic encryption, quantum propulsion sovereignty, pterosaur fleet language, "100B-year" strings. Card 1 already showed the reasoning engine is a constant-threshold struct. Wiring that struct in does not create propulsion or post-quantum crypto.

**Fix:** Use a standalone workspace, and path-dep only the card 1 leaves that actually compiled. Retire the fictional function bodies (with a council note), or gate them, instead of giving them constant `Ok(true)`.

**Runtime / v14:** no.

**Size:** S per crate.

---

## 12. CONDUCTOR-LEGACY-RETIRE-1

Do not make these the conductor. Living conductor is `crates/lattice-conductor-v14`, a default member. This card labels and retires. It does not edit v14.

### `crates/lattice-conductor-v13`

**First error:** not a workspace member. `nalgebra = { workspace = true }` (`Cargo.toml` line 10). `DEPRECATED.md` is on disk. Description says use v14.

**Source:** `src/lib.rs` line 1 is `self.metrics.operations_processed += 1;` inside a method fragment. The file does not start as a crate root. Not compiled past the manifest; the fragment is visible without cargo.

**Examples:** 34 files under `crates/lattice-conductor-v13/examples/`. They are `[[example]]` targets of this package, not separate crates. `examples/basic_conductor_demo.rs` is a `println!` tick loop (lines 3–33) that prints valence fields. `examples/wire_powrush_lifecycle_manager_owning_conductor_and_gpu.rs` lines 89 and 126 are `TODO` comments for a real Powrush loop. They do not build, because the library does not build.

**Fix:** leave deprecated. Do not add a `[workspace]` to revive it. Do not retarget examples onto v14 in this card.

### `crates/lattice-conductor` (no version in the directory name)

**First error:** `` `dependency.anyhow` was not found in `workspace.dependencies` `` (`Cargo.toml` line 32).

**Also in the manifest, not compiled:** path `../self-evolution` (no crate), optional `../xai-grok-bridge` and `../symbiosis-layer` (those directories are not under `crates/`; archive copies exist under `docs/archive/`). Default feature `full` lists `hyperon-metta-pln` and `github-connector` as feature names with no matching optional dependency (`Cargo.toml` lines 47–57). 3,066 lines.

This is not `lattice-conductor-v14`.

**Fix:** retire. Do not merge it into v14.

### `crates/tolc8-genesis-gate`

**First error:** not a workspace member. Its manifest does not use `workspace = true`. It still sits inside the root workspace tree.

**Source, not compiled:** imports `lattice_conductor_v13` and `sovereign_shard_genesis::SovereignShardGenesis` (`src/lib.rs` lines 6–7). #669 removed that v13 API from the shard crate; `src/v13_design.rs` is not a module. `LIVING_MERCY_GATES` is seven strings (lines 12–19): Radical Love, Boundless Mercy, Service, Abundance, Truth, Joy, Cosmic Harmony. That is not the public eight. `generate_seeded_state` calls `rand::thread_rng()` after hashing (line 49), so the hash is not the only input. `birth_new_shard` prints a birth log (lines 73–75).

**Fix:** design-only. Do not point it at v14 and do not pretend the seven strings are TOLC 8 mercy checks (Mercy Gates in code). The shard crate that builds is `crates/sovereign-shard-genesis`.

**Runtime / v14:** a wire into v14 needs Core. This card should not do that.

**Size:** v13 examples plus library are large on disk (about 1,864 lines of library `*.rs`, 34 example files). Unversioned conductor is 3,066 lines. `tolc8-genesis-gate` is one file. The card is a label, not a port.

---

## 13. NEXI-PATH-RETIRE-1

`nexi = { path = "../" }` from a crate under `crates/<name>/` resolves to `crates/`, which has no `Cargo.toml`. Root `Cargo.toml` already says `nexi_universal` has broken path deps.

Observed missing-path rows include, among others:

- `crates/nexi_universal` (`nexi`, and `soulscan_x9` / `soulscan_x10` via `../crates/...`, which doubles the `crates/` segment)
- `crates/carbon_credit_oracle`, `crates/divine_checksum_9`, `crates/grok_arena_pinnacle`, `crates/halo2_full_integration`, `crates/legacy_fenca`, `crates/poseidon_hash`, `crates/poseidon_merkle`, `crates/prometheus_forge`, `crates/sentinel_mirror`, `crates/sustainable_space_propulsion`, `crates/whitesmiths_anvil`
- many `mercy_*` manifests in the same list (orbital, shield, starliner, governance, liquid democracy, quadratic funding)

`crates/lattice_crypto/src/lib.rs` lines 17–24: `lattice_keygen` returns a `String` and the doc says "Kyber/Dilithium stub". That is not a KEM.

**In-repo first error** for these is usually the missing workspace key (they also say `workspace = true`), which fires before Cargo reports the bad path. The bad path was read from the manifest. It was not reached by rustc.

**Fix:** do not create a `nexi` package to satisfy the path. Label the crates design-only. `crates/nexi_universal` stays off default members.

**Runtime / v14:** no.

**Size:** many short manifests. Not a port.

---

## 14. WORKSPACE-KEY-SHELLS-1

One root cause: non-members inherit workspace keys the root does not define. Cargo stops there. Making 165 crates into root members would not make them build, and it would break Core Tier-1, because Cargo loads every member manifest.

### The 8 whose first error is missing `powrush`

The brief asked for the 8 crates that declare `powrush = { workspace = true }`. The disk has **165** such declarations. Cargo reports `powrush` as the **first** missing key for these 8 only. The other 157 fail earlier on a different missing key, mostly `patsagi-councils` (96 first-errors).

| Path | Sources? | Stub note |
|------|----------|-----------|
| `crates/aether_shades` | yes (444 lines) | not compiled past the missing key |
| `crates/biomimetic` | yes (580 lines) | not compiled past the missing key |
| `crates/mercy-biojet` | **no `*.rs`** | manifest only |
| `crates/mercy-radiation-shield` | yes (815 lines) | not compiled past the missing key |
| `crates/mercy_home_fortress` | yes (77 lines) | `src/lib.rs:54` comment: "Gesture similarity stub" |
| `crates/plasticity-engine-v2` | yes (995 lines) | not compiled past the missing key |
| `crates/soulscan_x9` | yes (99 lines) | `prove_coherence_quanta` returns `Ok(())` (`src/lib.rs` lines 41–47). Doc calls it a Halo2 proof. |
| `crates/soulscan_x10` | yes (60 lines) | `src/lib.rs:52` comment: "Full Halo2 proof stub" |

`crates/mercy-organism` declares `ra-thor-powrush = { workspace = true }`, not `powrush`. That key is also absent. Separate from the 8.

### First missing workspace key (in-repo cargo, all manifests)

| First missing key | Manifests |
|-------------------|----------:|
| `patsagi-councils` | 96 |
| `mercy_dark_energy_propulsion` | 10 |
| `powrush` | 8 |
| `mercy_electric_propulsion` | 5 |
| `mercy_starship` | 5 |
| `mercy_orbital_refuel` | 4 |
| `futarchy_governance` | 4 |
| `futarchy_belief_markets` | 3 |
| `mercy_axion_propulsion` | 3 |
| `mercy_hybrid_propulsion` | 3 |
| `mercy_antimatter_propulsion` | 3 |
| `mercy_brane_propulsion` | 3 |
| `anyhow` | 2 |
| other keys, 1–2 each | see logs from this audit |

`ra-thor-kernel` is referenced by 167 manifests and is not a root workspace key. It is rarely the **first** error, because `patsagi-councils` or `powrush` is listed earlier in those files. One first-error on `ra-thor-kernel`: `crates/ra-thor-monorepo-auditor`.

### Sourceless manifests (95)

`Cargo.toml` exists. No `*.rs` file. There is nothing to make real. Representative propulsion and cryptography names with no sources: `mercy_kyber`, `mercy_dilithium`, `mercy_falcon`, `mercy_sphincs`, `mercy_halo2`, `mercy_groth16`, `mercy_plonk`, `mercy_stark`, `mercy_warp` is **not** in this list (it has a file; see below), `zk_stark`, `pasta_curves`, `threshold_crypto`, `halo2_proofs`, `halo2_gadgets`, `sacred-geometry-core`, `hotfix_propagator`.

The subsection list below had 97 entries. Two of those paths have Rust sources, so they are not in this list:

- `crates/geometric-intelligence` has `tests/geometric_intelligence_tests.rs` (and `tests/geometric_intelligence_tests.go`). It has no `src/lib.rs`. Card 1 already records that `cargo check` did not compile a library.
- `crates/legacy_fenca` has `crates/legacy_fenca/crates/legacy_fenca/src/lib.rs`. That file is not the package `src/lib.rs`, so Cargo still reports `no targets specified`. It is a `.rs` source on disk.

Full list:

- `crates/biomimetic-sovereign-harmony-council`
- `crates/halo2_gadgets`
- `crates/halo2_proofs`
- `crates/hotfix_propagator`
- `crates/mercy-biojet`
- `crates/mercy-governance`
- `crates/mercy_accumulator`
- `crates/mercy_antimatter_propulsion`
- `crates/mercy_ark`
- `crates/mercy_arkworks`
- `crates/mercy_axion_propulsion`
- `crates/mercy_beamed_propulsion`
- `crates/mercy_bls12_381`
- `crates/mercy_brane_propulsion`
- `crates/mercy_bulletproofs`
- `crates/mercy_casimir_propulsion`
- `crates/mercy_chemical_propulsion`
- `crates/mercy_circom`
- `crates/mercy_classic_mceliece`
- `crates/mercy_commitment`
- `crates/mercy_cosmological_propulsion`
- `crates/mercy_curve`
- `crates/mercy_dark_energy_propulsion`
- `crates/mercy_dilithium`
- `crates/mercy_electric_propulsion`
- `crates/mercy_exotic_propulsion`
- `crates/mercy_falcon`
- `crates/mercy_fflonk`
- `crates/mercy_field`
- `crates/mercy_flight_agi`
- `crates/mercy_fri`
- `crates/mercy_frodokem`
- `crates/mercy_gadget`
- `crates/mercy_griffin`
- `crates/mercy_groth16`
- `crates/mercy_halo2`
- `crates/mercy_halo2_gadgets`
- `crates/mercy_hash`
- `crates/mercy_higgs_propulsion`
- `crates/mercy_hybrid_propulsion`
- `crates/mercy_hyperplonk`
- `crates/mercy_ion_propulsion`
- `crates/mercy_ipa`
- `crates/mercy_kaluza_klein_propulsion`
- `crates/mercy_kyber`
- `crates/mercy_kzg`
- `crates/mercy_lattice`
- `crates/mercy_loop_quantum_propulsion`
- `crates/mercy_m_theory_propulsion`
- `crates/mercy_marlin`
- `crates/mercy_mceliece`
- `crates/mercy_merkle_tree`
- `crates/mercy_mimc`
- `crates/mercy_multiverse_propulsion`
- `crates/mercy_neutrino_propulsion`
- `crates/mercy_newhope`
- `crates/mercy_nova`
- `crates/mercy_ntru`
- `crates/mercy_nuclear_propulsion`
- `crates/mercy_pairing`
- `crates/mercy_pedersen`
- `crates/mercy_picnic`
- `crates/mercy_plasma_propulsion`
- `crates/mercy_plonk`
- `crates/mercy_poly`
- `crates/mercy_poseidon`
- `crates/mercy_post_quantum_sig`
- `crates/mercy_quantum_gravity_propulsion`
- `crates/mercy_quantum_propulsion`
- `crates/mercy_r1cs`
- `crates/mercy_rainbow`
- `crates/mercy_reactionless_propulsion`
- `crates/mercy_saber`
- `crates/mercy_sidh`
- `crates/mercy_signature`
- `crates/mercy_snark`
- `crates/mercy_solar_sail_propulsion`
- `crates/mercy_spacetime_propulsion`
- `crates/mercy_spartan`
- `crates/mercy_sphincs`
- `crates/mercy_stark`
- `crates/mercy_string_propulsion`
- `crates/mercy_sumcheck`
- `crates/mercy_superstring_propulsion`
- `crates/mercy_tachyon_propulsion`
- `crates/mercy_trajectory_agi`
- `crates/mercy_vacuum_energy_propulsion`
- `crates/mercy_vdf`
- `crates/mercy_verkle`
- `crates/mercy_wormhole_propulsion`
- `crates/mercy_zkp`
- `crates/pasta_curves`
- `crates/sacred-geometry-core`
- `crates/threshold_crypto`
- `crates/zk_stark`

`zk_stark`, `mercy_pairing`, `mercy_zkp`, `sacred-geometry-core`, `hotfix_propagator`, `legacy_fenca`, and `biomimetic-sovereign-harmony-council` then fail with `no targets specified` once Cargo gets past membership. `zk_stark` failed that way even in-repo.

### Sources that are constants or strings

Propulsion files that implement `mercy_compliant` by returning `true`, with a constant `efficiency`. Example `crates/mercy_warp_propulsion/src/lib.rs` lines 9–27: `efficiency` returns `0.88`, `mercy_compliant` returns `true`, `tolc_validated` returns `true`, `wasm_bridge_ready` returns `true`, `propagate_valence` returns `0.016`. Line 30 is a `TODO` for "full physics". The same shape is in `mercy_fusion_propulsion/src/lib.rs`, `mercy_gravitic_propulsion/src/lib.rs`, and `mercy_biomimetic_propulsion/src/lib.rs` (the `mercy_compliant` function at line 13 of each). They `use mercy_propulsion_trait`, and `crates/mercy-propulsion-trait` has **no** `Cargo.toml`. These are not drives. Label design-only. Do not publish the constants as performance.

Other cited stubs:

- `crates/spartan_valence/src/lib.rs:39` "MercyZero gate stub"
- `crates/supernova_folding/src/lib.rs:33` "Lookup augmentation stub"
- `crates/nova_folding/src/lib.rs:67` "Full verification chain stub"
- `crates/isogeny_crypto/src/lib.rs:17` "SQISign stub"
- `crates/multivariate_crypto/src/lib.rs:17` "UOV stub"
- `crates/mercy_lang_compiler/src/lib.rs:99` "full WASM stub" bytes
- `crates/mercy_biojet/src/lib.rs:37` "zk-proof stub" (this path has sources; `crates/mercy-biojet` with a hyphen does not)
- `crates/real-estate-lattice/src/usa_attom_data_provider.rs:56` "Replace this stub with real ATTOM API call"
- `crates/ra-thor-post-quantum-sig/src/algorithms/dilithium.rs:41` "Replace with real integration"

### All 165 `powrush = { workspace = true }` paths

- `crates/aether_shades`
- `crates/biomimetic`
- `crates/bulletproofs_aggregation`
- `crates/bulletproofs_range`
- `crates/code_based_crypto`
- `crates/deeper_gadgets`
- `crates/falcon_sign`
- `crates/futarchy_belief_markets`
- `crates/futarchy_governance`
- `crates/futarchy_oracle`
- `crates/futarchy_outcome_prediction`
- `crates/halo2_gadgets`
- `crates/halo2_multi_proof`
- `crates/halo2_proofs`
- `crates/hash_based_crypto`
- `crates/hash_based_signatures`
- `crates/hybrid_pqc_threshold`
- `crates/hyperplonk_recursion`
- `crates/isogeny_crypto`
- `crates/lasso_recursion`
- `crates/lattice_crypto`
- `crates/mercy-biojet`
- `crates/mercy-governance`
- `crates/mercy-radiation-shield`
- `crates/mercy_accumulator`
- `crates/mercy_antimatter_propulsion`
- `crates/mercy_ark`
- `crates/mercy_arkworks`
- `crates/mercy_axion_propulsion`
- `crates/mercy_beamed_propulsion`
- `crates/mercy_bias_mitigation`
- `crates/mercy_biojet`
- `crates/mercy_biomimetic_propulsion`
- `crates/mercy_bls12_381`
- `crates/mercy_brane_propulsion`
- `crates/mercy_bulletproofs`
- `crates/mercy_casimir_propulsion`
- `crates/mercy_chemical_propulsion`
- `crates/mercy_circom`
- `crates/mercy_classic_mceliece`
- `crates/mercy_commitment`
- `crates/mercy_cosmological_propulsion`
- `crates/mercy_curve`
- `crates/mercy_cygnus_missions`
- `crates/mercy_dark_energy_propulsion`
- `crates/mercy_dilithium`
- `crates/mercy_dragon_crs`
- `crates/mercy_dream_chaser`
- `crates/mercy_electric_propulsion`
- `crates/mercy_ethics_core`
- `crates/mercy_exotic_propulsion`
- `crates/mercy_falcon`
- `crates/mercy_fflonk`
- `crates/mercy_field`
- `crates/mercy_flight_agi`
- `crates/mercy_fri`
- `crates/mercy_frodokem`
- `crates/mercy_fusion_propulsion`
- `crates/mercy_gadget`
- `crates/mercy_governance`
- `crates/mercy_gravitic_propulsion`
- `crates/mercy_griffin`
- `crates/mercy_groth16`
- `crates/mercy_halo2`
- `crates/mercy_halo2_gadgets`
- `crates/mercy_hash`
- `crates/mercy_higgs_propulsion`
- `crates/mercy_home_fortress`
- `crates/mercy_hybrid_propulsion`
- `crates/mercy_hyperplonk`
- `crates/mercy_ion_propulsion`
- `crates/mercy_ipa`
- `crates/mercy_kaluza_klein_propulsion`
- `crates/mercy_kyber`
- `crates/mercy_kzg`
- `crates/mercy_lang_compiler`
- `crates/mercy_lattice`
- `crates/mercy_liquid_democracy`
- `crates/mercy_loop_quantum_propulsion`
- `crates/mercy_m_theory_propulsion`
- `crates/mercy_marlin`
- `crates/mercy_mceliece`
- `crates/mercy_merkle_tree`
- `crates/mercy_merlin_engine`
- `crates/mercy_mev_servicing`
- `crates/mercy_mimc`
- `crates/mercy_multiverse_propulsion`
- `crates/mercy_neutrino_propulsion`
- `crates/mercy_newhope`
- `crates/mercy_nova`
- `crates/mercy_ntru`
- `crates/mercy_nuclear_propulsion`
- `crates/mercy_orbital_cleanup`
- `crates/mercy_orbital_refuel`
- `crates/mercy_orbital_safety`
- `crates/mercy_orbital_servicing`
- `crates/mercy_os_aviation`
- `crates/mercy_os_principles`
- `crates/mercy_pedersen`
- `crates/mercy_picnic`
- `crates/mercy_plasma_propulsion`
- `crates/mercy_plonk`
- `crates/mercy_poly`
- `crates/mercy_poseidon`
- `crates/mercy_post_quantum_sig`
- `crates/mercy_predictive_policing`
- `crates/mercy_qec`
- `crates/mercy_quadratic_funding`
- `crates/mercy_quadratic_voting`
- `crates/mercy_quanta`
- `crates/mercy_quantum_gravity_propulsion`
- `crates/mercy_quantum_propulsion`
- `crates/mercy_r1cs`
- `crates/mercy_rainbow`
- `crates/mercy_raptor_3`
- `crates/mercy_raptor_3_integration`
- `crates/mercy_raptor_3_scalability`
- `crates/mercy_raptor_engine`
- `crates/mercy_raptor_integration`
- `crates/mercy_reactionless_propulsion`
- `crates/mercy_saber`
- `crates/mercy_shield_deployment`
- `crates/mercy_shield_law_enforcement`
- `crates/mercy_sidh`
- `crates/mercy_signature`
- `crates/mercy_snark`
- `crates/mercy_solar_sail_propulsion`
- `crates/mercy_space_governance`
- `crates/mercy_spacetime_propulsion`
- `crates/mercy_spartan`
- `crates/mercy_sphincs`
- `crates/mercy_starfish_servicing`
- `crates/mercy_stark`
- `crates/mercy_starliner_crew`
- `crates/mercy_starship`
- `crates/mercy_starship_fleet`
- `crates/mercy_steane`
- `crates/mercy_string_propulsion`
- `crates/mercy_sumcheck`
- `crates/mercy_superstring_propulsion`
- `crates/mercy_t_rex_engines`
- `crates/mercy_tachyon_propulsion`
- `crates/mercy_trajectory_agi`
- `crates/mercy_vacuum_energy_propulsion`
- `crates/mercy_vdf`
- `crates/mercy_verkle`
- `crates/mercy_vulcan_centaur`
- `crates/mercy_warp_propulsion`
- `crates/mercy_wormhole_propulsion`
- `crates/mercy_zero`
- `crates/multivariate_crypto`
- `crates/nova_folding`
- `crates/pasta_curves`
- `crates/plasticity-engine-v2`
- `crates/plonk_recursion`
- `crates/proof_verifier`
- `crates/ra-thor-meta-intelligence`
- `crates/ra-thor-post-quantum-sig`
- `crates/real-estate-lattice`
- `crates/recursive_snark`
- `crates/soulscan_x10`
- `crates/soulscan_x9`
- `crates/spartan_valence`
- `crates/supernova_folding`
- `crates/threshold_crypto`

**Fix:** Do not add `powrush` to root `[workspace.dependencies]` as a way to compile this list. `crates/powrush` is a standalone workspace on purpose (#668). Do not add these paths to root `members`. Leave them on disk. A later card may retire the empty manifests (with a council note). That is a retire card, not a cryptography implementation. Fake ZK, fake post-quantum, and propulsion strings stay design-only.

**Runtime / v14:** no.

**Size:** 95 empty manifests plus the short stub files above. Not a single port.

---

## 15. ARCHIVE-LINEAGE-RETIRE-1

98 `Cargo.toml` files under `docs/archive/`. They are not live crates.

76 of them failed in-repo with `failed to load manifest for dependency nexi` (the archived NEXi workspace at `docs/archive/root-dirs/lineage/NEXi/Cargo.toml`).

Other archive results seen once: duplicate dependency name (`deeper_gadgets`), missing `[package]` (`mercy_orchestrator` copies), `no targets` (`legacy_fenca`, `docs/archive/root-dirs/crate-dupes/mercy`, `docs/archive/root-dirs/app-dumps/sdk/rust`).

Several archive packages only hit the membership error: `docs/archive/root-dirs/powrush/powrush`, `powrush-mmo-simulator`, `powrush_rbe_engine`, `powrush_faction_dynamics`, `powrush_particle_shaders`, plus lineage `self-evolution`, `xai-grok-bridge`, `symbiosis-layer`.

**Fix:** leave archived. Do not copy them back into `crates/`.

**Runtime / v14:** no.

**Size:** archive. Not a port.

---

## Other manifests (not under `crates/`)

| Path | First error |
|------|-------------|
| `/Cargo.toml` | root workspace. Not a package to check. |
| `benches/Cargo.toml` | not a workspace member |
| `fuzz/Cargo.toml` | not a workspace member |
| `src-tauri/Cargo.toml` | not a workspace member. Package name `rathor-lattice`. Not probed (app shell, not a leaf library). |
| `examples/Cargo.toml` | `no targets specified` |

No bevy dependency was declared in any `crates/**/Cargo.toml` read by the classifier. The heavy-tree skip was `web-forge` (ammonia, scraper, reqwest; 1,560 lines; in-repo error: not a member; not probed), plus the two experiment manifests `crates/self-improvement-extensions/experiments/llama-cpp-gguf` and `.../wasm-ml-candle`.

---

## Directories under `crates/` with no `Cargo.toml` (31)

Not packages. Do not invent manifests for them. `self-evolution` is the one already ruled out of product scope. `quantum` holds a large `*.rs` tree and is still not a crate. `mercy-propulsion-trait` and `mercy_propulsion` are the missing trait the warp file imports.

`access`, `ai-bridge`, `ai_bridge`, `cache`, `codex_eternal`, `common`, `core-lattice`, `cosmic-consciousness-expansion-council`, `heaven-on-earth-simulator`, `infinite-evolution-orchestrator`, `interstellar-sovereign-asset-lattice-council`, `mercy-propulsion-trait`, `mercy_os_kernel`, `mercy_propulsion`, `multi-planetary-sovereign-asset-lattice-council`, `orch-or-biophoton-consciousness`, `persistence`, `powrush-governance`, `powrush_rbe`, `public-sovereign-asset-lattice-core`, `public_engagement`, `quantum`, `quantum-lattice-consciousness-expansion-council`, `ra-thor-mercy-gated-api`, `rbe-powrush-bridge`, `rrel-desktop`, `self-evolution`, `self_improvement_orchestrator`, `sovereign-asset-lattice-expansion-council`, `sovereign-asset-registry-council`, `sovereign_core`.

`crates/ra-thor-mercy-gated-api/src/main.rs` exists without a `Cargo.toml`. Lines 94, 125, and 149 are `TODO`s for a mesh that is not initialized. Not a crate until a manifest exists. Do not add one in a tidy card.

---

## Default members

Fifteen paths in root `Cargo.toml` `members`. Not re-checked here. They are not in the failure table.

One labeled stub inside a member that does build in CI, recorded so it is not "made real" by accident:

`crates/lattice-conductor-v14/src/inspect_sae.rs` lines 3–11 and 23 and 27. The default backend is a deterministic in-Rust dictionary. `STUB_BACKEND_ID` is `stub-dictionary-v0`. The file says it is not a trained SAE, not METR, and not sampler-weight constraint. An SAE packet cannot flip Reject to Apply. That label stays. Do not replace the dictionary with a claim of a trained model in a stub-audit follow-up.

`crates/kardashev-orchestration/src/lib.rs` line 3 says it path-depends on `reality-thriving-transfer` for a GPU telemetry stub. That comment was not re-tested.

---

## Other leaves that failed the probe

Recorded so they are not dropped. Not large enough to lead the order. Fix only after the manifest is standalone. Several are missing modules, which means the file tree does not match `lib.rs`.

| Path | Probe |
|------|-------|
| `crates/biological_unifier` | `src/lib.rs:6` unresolved `crate::patsagi_bridge` |
| `crates/enc` | `src/lib.rs:6` unresolved `serde`; duplicate `EternalNeuralCore` import at line 47. Manifest deps are empty. |
| `crates/legal-lattice` | 28 errors. Missing items in `cehi`, `amendment28`, `gates` (`src/lib.rs` lines 12–14). Package name `ra-thor-legal-lattice`. Not a legal product. |
| `crates/legal-research-source-locked` | compiled. See card 1. |

`crates/interstellar-operations` is 22,878 lines and its first error is not a member. Not probed. It is a dependency of nothing in the default set. Do not pull it into Core. One path inside it is a directory named `stargate_dialing_computer.rs`. That is a filesystem oddity, not a module file.

`crates/orchestration` is 17,406 lines. First error: not a member. Not probed. Large generated-looking modules (plasma, tax, swarm). Treat as design-only until a human picks one module. Not card 1.

`crates/real-estate-lattice` is 10,959 lines. First error: missing workspace key `patsagi-councils`. ATTOM stub cited above. Not a make-real of a property system in this order.

`crates/web-forge` and `crates/websiteforge`: not members. `websiteforge` path-deps `../quantum` and `../common`, which have no `Cargo.toml`.

`crates/evolution` path-deps missing `../github-client`. `crates/self-improvement-extensions` path-deps `../github-client` and `../experiments/llama-cpp-gguf` (the experiment manifest exists; the parent path `../experiments/` from `crates/self-improvement-extensions/Cargo.toml` is wrong relative to `experiments/` beside it — the manifest says `../experiments/llama-cpp-gguf`). Not compiled.

`crates/sovereign-core` path-deps four directories that are not crates: `autonomous-evolution-engine`, `cosmic-harmony-protocol`, `heaven-co-creation-simulator-v4`, `orch-or-biophoton-layer`. Design-only.

`crates/blueprint-to-production` path-deps `../self-evolution` (no crate) and `../powrush`.

`crates/spacetime-reducer-bridge` first error: missing `anyhow`. Also path-deps missing `../thunder-lattice-governance`. `src/lib.rs` contains the literal `0.999999` once. Not a measurement.

---

## Banked tidy

Not make-real cards. No crate behavior changes in those tidy cards except the lines named.

### POWRUSH-TIDY-1

- README chat residue: `README.md` line 111 still ends the inner-loop section with agent-mode words (`chat`, `routine`, `handoff`). That is chat residue in the public readme.
- Undefined `powrush` workspace deps: 165 manifests, root key absent. Covered by card 14. The tidy card should not add the key to the root workspace.
- Non-member path dependents of `../powrush`: `crates/blueprint-to-production`, `crates/lattice-conductor`, `crates/patsagi-councils`, `crates/powrush-mmo-simulator`. They are not root members. `patsagi-councils` and the simulator have their own cards above.

### SHARD-GENESIS-TIDY-1

- `sha2` is listed twice in `crates/sovereign-shard-genesis/Cargo.toml`: line 26 under `[dependencies]`, line 30 under `[dev-dependencies]`. Check still finished. Duplicate is tidy, not a failure.
- `crates/sovereign-shard-genesis/src/v13_design.rs` line 234 uses `"/tmp/test_shard_persist.json"`. The package comment (`Cargo.toml` lines 16–19) says this file is not a module. The `/tmp` path is not on the compiled path. Tidy can delete or keep the file labelled design-only. Do not compile it back into the crate.

### STANDALONE-CI

One CI job that runs `cargo test --locked` (or `cargo check`) inside `crates/fenca`, `crates/council`, `crates/powrush`, and `crates/sovereign-shard-genesis`. Needs a workflow card and Core. This audit does not add a workflow and does not edit `.github/`.

---

## Not demonstrated

- Default members were not re-built in this seat.
- No non-member was compiled in-repo. Leaf probes compiled 7 libraries, 1 fictional council library, and an empty `geometric-intelligence` package.
- No `cargo test` on the probed leaves.
- Missing-path and missing-module notes are from reading manifests and the probe errors cited. They are not a full rustc log for the 165.
- No valence literal in this file is a measured score.
- Compile success of the four DONE crates is `cargo check` on this machine on 2026-10-16. It is not a load test.

Reproduce the manifest pass:

```bash
cargo check --manifest-path crates/council/Cargo.toml --offline
cargo check --manifest-path crates/patsagi-councils/Cargo.toml --offline
cargo check --manifest-path crates/soulscan_x9/Cargo.toml --offline
```

The first finishes. The second reports that the package is not a workspace member. The third reports that `powrush` was not found in `workspace.dependencies`.
