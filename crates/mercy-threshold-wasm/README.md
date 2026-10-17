# mercy-threshold-wasm

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member. `mial` is a relative path dependency. The default feature is `native`. Toolchain: `rust-version` is 1.89, the minimum in this crate's `Cargo.lock` (`uuid` 1.28.0 requires 1.89.0).

`MercyThresholdBridge::new` compiles the bytes you pass and looks up export names. Empty bytes return an error. This crate does not embed a proof. When a score export is missing, that method returns the source literal `0.0`. That `0.0` is not a measured score.

`native::evaluate_geometry_mercy_hybrid` calls `mial::mwpo`. A Platonic solid with order `0`, density `0`, and step `0` has resonance `0.96` and `truth` `0.999`. `valence` below the source literal `0.999999` returns an error.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2

## Verify

```bash
cargo build --manifest-path crates/mercy-threshold-wasm/Cargo.toml --locked
cargo test --manifest-path crates/mercy-threshold-wasm/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/mercy-threshold-wasm/Cargo.toml --locked --all-targets -- -D warnings
```
