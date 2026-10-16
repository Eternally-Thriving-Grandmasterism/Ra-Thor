# mercy_gating_runtime

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member.

`race_gate_amplifier` returns the match-arm literals in `src/lib.rs` (for example Sovereign on `one_organism_unity` is `1.38`). `gate_17_24_passes` compares a score with the thresholds in that function (`one_organism_unity` is `0.90`). Those numbers are source literals, not measured results.

`GateThresholdMap::set_threshold` returns Err when `new_value` is non-finite or below the stored value, and leaves that value in place. An equal or higher finite value is stored. `new` inserts the literals in that function (`council_harmony` is `0.80`).

`MercyGatingRuntime::new` sets `ma_at_threshold` to `717.0`. `apply_council_tuning` on `MaAtThreshold` and `GateThreshold` refuses `new_value` when it is non-finite or below the stored value (`success` is false) and leaves that value in place. An equal or higher finite `new_value` is stored. A missing name reports `0.75` from `get_gate_threshold`. `0.75` is that function's source literal, not a measured floor. `RaceAmplifier` does not change a threshold. `success` is true for that target. `set_gate_threshold` returns `Err` on the same test and does not use a `0.5` floor. `apply_council_tuning` does not call it.

`ma_at_threshold` and `gate_thresholds` are public fields, so the refuse-a-decrease guarantee covers the methods only, and a direct field write bypasses it.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1

## Council note

`src/error.rs` does not exist. `MercyError` is not defined. These files stay on disk and are not modules: `dynamic_tuning.rs`, `gates_17_24_enforcement.rs` (`MaAtScore` is not defined here), `hot_reload.rs`, `metrics.rs`, `patsagi_arbitration.rs`, `patsagi_governance.rs`, `runtime.rs`. `src/bin/metrics_server.rs` is not a binary target. The `0.82` in that file is a source literal in sample JSON, not a measured success rate.

## Verify

```bash
cargo build --manifest-path crates/mercy_gating_runtime/Cargo.toml --locked
cargo test --manifest-path crates/mercy_gating_runtime/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/mercy_gating_runtime/Cargo.toml --locked --all-targets -- -D warnings
```
