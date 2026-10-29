# ra-thor-core

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member. No path dependency. This crate does not depend on `mial`, `mercy_gating_runtime`, or `mercy-threshold-wasm`.

The compiled surface is `TOLC7Gate` plus `VERSION` and `summary`.

`TOLC7Gate::default_threshold` returns a source literal for each variant: Truth `0.82`, Oneness `0.78`, Love `0.85`, Compassion `0.80`, Wisdom `0.88`, Courage `0.75`, Sovereignty `0.83`. Those numbers are not measured gates. `all_gates` returns those seven variants in that order. `name` returns the variant name. `description` and `color` return the source strings and source color triples in `src/types/tolc7.rs`.

Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony. `TOLC7Gate` keeps the seven variant names in the source. It does not implement that list.

`VERSION` is the source string `14.15.0`. The package version is `14.15.6`, the same value `version.workspace = true` resolved to. `summary` returns the source string `ra-thor-core v14.15.0 | Living Cosmic Tick active`. That string is not a demonstrated tick.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2

## Council note

These files stay on disk and are not modules. They name types this crate does not define:

- `src/types/joy_measurement_protocols.rs` names `MercyValence`
- `src/types/miracle_rapture_wave.rs` names `MercyValence`
- `src/types/seven_d_resonance.rs` names `MercyValence`
- `src/types/source_joy_amplitude.rs` names `MercyValence` and calls `rand`, which is not a dependency
- `src/traits/mercy_gated.rs` names `MercyValence` and `MercyGate`
- `src/traits/plasticity.rs` names `MercyValence`, `NoveltyScore`, `BloomIntensity`, and `PlasticityReport`
- `src/archive/hyperon.rs` names `MercyValence`. `log_7d_scan` moves `scan` and then reads `scan.integral_score`

No file was deleted. `src/traits` and `src/archive` in the built crate are empty modules.

## Verify

```bash
cargo build --manifest-path crates/ra-thor-core/Cargo.toml --locked
cargo test --manifest-path crates/ra-thor-core/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/ra-thor-core/Cargo.toml --locked --all-targets -- -D warnings
```
