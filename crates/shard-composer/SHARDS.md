# Sovereign Shard Profiles

This document describes the available shard compositions and persistence security configuration.

## Environment Variable Configuration

The persisted state of `ShardComposerAdapter` (used for self-evolution and epigenetic blessings) is protected with HMAC-SHA256.

You can override the base key material using the following environment variable:

```bash
export RA_THOR_HMAC_BASE_KEY="your-strong-secret-here"
```

### Behavior

- If `RA_THOR_HMAC_BASE_KEY` is set, it will be used as the base material for HKDF key derivation.
- If not set, a default base material is used.
- The final HMAC key is always derived using **HKDF-SHA256**.

### Security Notes

- Changing this value will invalidate previously signed state files (they will be rejected on load).
- Use a strong, unique value in production or sensitive environments.
- This variable only affects the `shard-composer` persistence layer.

## Profiles

This package does not define cargo features named `full`, `focused-real-estate`, `real-estate`, `focused-geometry`, or `geometry`.

`geometric-intelligence` has no library. `real-estate-lattice` is not a standalone build. `crates/mercy` is a root member. Those paths are not dependencies. `quantum-swarm-orchestrator` has no `full` feature. `patsagi-councils` has no `full` feature.

The compiled library is `ShardComposerAdapter` on `ra-thor-quantum-swarm-orchestrator`, plus a re-export of `patsagi-councils` as `patsagi`.

## Usage

From this directory:

```bash
cargo test --all-targets --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check
```

`cargo xtask build-shard` is not a target of this package.
