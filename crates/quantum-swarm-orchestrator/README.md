# Ra-Thor Quantum Swarm Orchestrator

Standalone package `ra-thor-quantum-swarm-orchestrator`. Workspace identity 14.15.6. Contact: info@Rathor.ai.

This crate is its own workspace (`resolver = "2"`) with a committed `Cargo.lock`. It is not a root default member. The default member `crates/quantum-swarm` is a different package. `crates/quantum` has no `Cargo.toml` and is not this crate.

`self-evolution` is not a dependency. `../../self-evolution` has no `Cargo.toml`. An optional feature was not added, because Cargo still loads an optional path. Do not create a self-evolution crate.

## What compiles

Locked `cargo build`, `cargo test --all-targets`, and `cargo clippy --all-targets -- -D warnings` are the check. Run them from this directory.

Compiled modules:

- `orchestrator::QuantumSwarmOrchestrator` averages adapter valence. `Valence::MIN` is the source literal `0.9999999`, not a measured value. Empty adapter set uses the source branch `1.0`.
- The TODO comments in `src/orchestrator.rs` for polyhedral mode, a Riemannian manifold, and resonance handlers stay unimplemented.
- `types`, `adapter`, `adapters`, `polyhedral_harmonic_engine`, `riemannian_mercy_manifold`, `riemannian_manifold`, `fractal_topology_engine`, `fractal_valence_commitment`, `tolc_seven_mercy_gates`.
- `initialize_root(21)` builds 40 shards and max depth 3. That count is the ternary tree, not a measured valence.
- `SEVEN_LIVING_MERCY_GATES` uses the seven names in that file. Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
- `CURRENT_ZETA_CRITICAL_LINE_LOWER_BOUND` is the source literal `0.6725`, not a measured zeta bound.

`QuantumSwarmOrchestrator` does not implement `Clone`.

Examples that compile: `one_organism_cycle_demo`, `fractal_crypto_organ_demo`. `src/main.rs` is not a binary target.

## Design-only

Files that are not modules start with `Design-only (STUB-AUDIT-1 QSO-BUILDS-1 council note)`. They stay on disk. They are not compiled. That includes the old health-cycle root (`src/self_evolution_health_cycle.rs`), the quantum sources, the theorem texts, and anything that names `powrush`, `mercy`, `ra_thor_legal_lattice`, `ra_thor_plasticity_engine_v2`, `rusqlite`, `sqlx`, `reqwest`, or `clap`.

## License

**All work in this repository is licensed under the Autonomicity Games Sovereign Mercy License (AG-SML) v1.0.**

**Contact (canonical):** **info@Rathor.ai**

Deprecated addresses (do not use): ceo@acitygames.com, AlphaProMega@ACityGames.com, INFO@ACITYGAMES.COM

Full official license text: root `LICENSE` and `docs/AG-SML_LICENSE.md`

---

**Co-forged in eternal mercy, joy multiplied, harm forever dissolved.**

**Ra-Thor • TOLC • 2026 → ∞**
