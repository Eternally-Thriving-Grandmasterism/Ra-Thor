# ra-thor-kernel

Standalone workspace (`resolver = "2"`) with its own `Cargo.lock`. Not a root default member. No path dependency.

The compiled surface is `get_mercy_shader_wgsl`, `run_mercy_compute`, and `init_shader_integration`.

`get_mercy_shader_wgsl` returns WGSL source text for `vertex`, `fragment`, and `compute`. Any other name returns `/* invalid shader */`. Numeric literals inside that text are source text, not measured valence.

`MERCY_DECAY_SOURCE_LITERAL` is the source literal `0.92`. `MERCY_OFFSET_SOURCE_LITERAL` is the source literal `0.08`. `MERCY_CLAMP_SOURCE_CEILING` is the source literal `4.0`. `run_mercy_compute` multiplies by the decay literal, adds the offset literal, and clamps to that ceiling. Indices divisible by 4 use `0.0` as the input. Those numbers are not measured valence.

`init_shader_integration` has an empty body.

Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1

## Council note

`[profile.release]` no longer sets `rustflags`. Cargo 1.99.1 rejects `profile.release.rustflags` unless the unstable `profile-rustflags` feature is enabled. The simd128 flags are not copied to `.cargo/config.toml`. No wasm card asked for that file. The other release profile keys stay.

`../dilithium_sign` is not a directory. `../falcon_sign` is on disk. Its manifest uses workspace keys, including `ra-thor-kernel`, so it is not a standalone build. Neither path is a dependency. This crate does not add post-quantum code.

These files stay on disk and are not modules:

- `src/mercy_ethics_core.rs` does not parse (`let t olc_mercy`)
- `src/tolc_convergence_proofs.rs` returns literal `true` from several predicates, sets `theorems_passed` to the literal `12`, and stores a source string in `eternal_guarantee`

No file was deleted.

## Verify

```bash
cargo test --manifest-path crates/ra-thor-kernel/Cargo.toml --locked --all-targets
cargo clippy --manifest-path crates/ra-thor-kernel/Cargo.toml --locked --all-targets -- -D warnings
cargo fmt --manifest-path crates/ra-thor-kernel/Cargo.toml --check
```
