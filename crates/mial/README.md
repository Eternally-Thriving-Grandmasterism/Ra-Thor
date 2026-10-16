# mial

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member. `mercy_gating_runtime` is a relative path dependency.

`MercyWeightedPreferenceOptimization::evaluate_geometry_mercy_component` scores a geometry with the formulas in `src/mwpo.rs`. A Platonic solid with order `0`, density `0`, and step `0` has resonance `0.96`. `truth` is `0.999` when `chiral` is false and `0.995` when it is true. `valence` below `0.999999` returns `GateViolation { gate: "valence" }`. `overall` below `0.92` returns `LowAlignment`. Those figures are source literals, not measured results. `active_gates` is not read. `optimize_preferences` stores `gated` as false. No compiled check sets it true.

`PathologyDetectionEngine::detect_pathologies` returns the first matching substring label. `trigger_automatic_recalibration` builds a `GateThreshold` proposal whose `new_value` is `(current_score * 1.08).min(0.97)` and applies it. `1.08` and `0.97` are source literals. `apply_council_tuning` keeps the stored value when `new_value` is lower, and this method returns `Err` in that case. A missing name reports `0.75`.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1

## Council note

`src/mial.rs`, `src/safety_harness.rs`, `src/lattice_introspection.rs`, `src/integration.rs`, and `src/mod.rs` stay on disk and are not modules. They call methods this crate does not define (`evaluate_proposal`, `check_gates`, `evaluate_trajectory`, and others). `examples/` and `tests/` are not cargo targets for the same reason.

## Verify

```bash
cargo build --manifest-path crates/mial/Cargo.toml --locked
cargo test --manifest-path crates/mial/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/mial/Cargo.toml --locked --all-targets -- -D warnings
```
