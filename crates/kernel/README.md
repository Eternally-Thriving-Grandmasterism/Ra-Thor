# kernel

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member. No path dependency. Package name is `kernel`. This package is not a root workspace key named `ra-thor-kernel`.

The compiled surface is `PACKAGE_VERSION_SOURCE` and `STANDALONE_BOUNDARY`.

`PACKAGE_VERSION_SOURCE` is the source string `14.15.0`, the package version in `Cargo.toml`. That string is not a readiness measurement.

`STANDALONE_BOUNDARY` is the source string `lattice-conductor-v14 is not a path dependency; v14 boot is not compiled`.

Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2

## Council note

`lattice-conductor-v14` is a root member. It does not build as its own workspace. This card does not edit v14 and does not path-depend on it. `enforce_cosmic_loop_on_boot` and `arbitration_rejects_disable` stay in `src/lattice_v14_boot.rs`, which is not a module.

These files stay on disk and are not modules:

- `lib.rs` (crate root, not `src/lib.rs`) declares modules that are not source files
- `src/absolute_truth_enforcer.rs` imports `crate::mercy` and `crate::common`, which this crate does not define, and assigns the source literal `0.9999999`
- `src/tolc_core_enforcer.rs` imports the same missing paths and assigns that same source literal
- `src/permanence_code_loop.rs` and `src/root_core_orchestrator.rs` import `crate::mercy`, `crate::common`, and `crate::quantum`, which this crate does not define
- `src/fenca.rs` — `is_verified` returns the literal `true`, and `compute_mermin_violation` applies `.max(2.0)`. The GHZ and Mermin comments describe a simulation this file does not run
- `src/meta_orchestrator.rs` imports `ra_thor_mercy` and `ra_thor_common`, which are not dependencies
- `src/sub_core_trait.rs` names `RequestPayload`, which is not defined in that file, and `pre_check` returns the literal `true`
- `src/wasm_boot.rs` imports crates that are not dependencies

No file was deleted. `mial`, `mercy_gating_runtime`, and `mercy-threshold-wasm` are not root members from this card.

## Verify

```bash
cargo test --manifest-path crates/kernel/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/kernel/Cargo.toml --locked --all-targets -- -D warnings
cargo fmt --manifest-path crates/kernel/Cargo.toml --check
```
