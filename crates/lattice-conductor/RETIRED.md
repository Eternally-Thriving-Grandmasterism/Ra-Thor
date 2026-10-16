# RETIRED

**Package:** `lattice-conductor` (no version in the directory name)
**Card:** STUB-AUDIT-1 CONDUCTOR-LEGACY-RETIRE-1
**Contact:** info@Rathor.ai

This directory is retired. It is not `crates/lattice-conductor-v14`. This card does not merge any file from this directory into v14. It does not add a `[workspace]` table. It does not make this crate a root workspace member. No `Cargo.lock` is added, because this package does not compile.

## Council note — broken manifest

`docs/STUB_AUDIT_1.md` records the first `cargo check` error as `` `dependency.anyhow` was not found in `workspace.dependencies` ``. `Cargo.toml` sets `anyhow = { workspace = true }`. Root `[workspace.dependencies]` has no `anyhow` key.

Also in that manifest, and not compiled:

- `ra-thor-self-evolution = { path = "../self-evolution" }`. `crates/self-evolution` has no `Cargo.toml`.
- Optional paths `../xai-grok-bridge` and `../symbiosis-layer`. Those directories are not under `crates/`. Archive copies exist under `docs/archive/`.
- Default feature `full` lists `hyperon-metta-pln` and `github-connector`. Those are phantom features: feature names with no matching optional dependency.

Every `.rs` file in this package starts with the design-only header for this card. The bytes under that header are the previous file. `src/lattice_conductor_v14.rs` is one of those files. It is not the living conductor crate.

The living conductor is `crates/lattice-conductor-v14`. This package stays a different crate.
