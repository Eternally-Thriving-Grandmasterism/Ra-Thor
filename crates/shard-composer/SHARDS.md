# Sovereign Shard Profiles

This document describes the available shard compositions and persistence security configuration.

## Environment Variable Configuration

The persisted state of `ShardComposerAdapter` is protected with HMAC-SHA256.

`RA_THOR_HMAC_BASE_KEY` is required. This crate does not contain a key.

```bash
export RA_THOR_HMAC_BASE_KEY="your-strong-secret-at-least-32-bytes"
```

### Behavior

- The variable's UTF-8 bytes are the HKDF input. They are not trimmed.
- The value must be set and at least 32 bytes. Unset, empty, non-UTF-8, or shorter than 32 bytes fails closed.
- `save_to_file` returns an `io::Error` whose message names `RA_THOR_HMAC_BASE_KEY`. It does not write the file.
- `load_from_file` does not accept a signed file in that case. It returns a fresh adapter. It does not panic.
- `save_to_file` refuses to overwrite an existing file that does not verify under the current base. The `io::Error` is `InvalidData`. The message says the existing file failed verification and must be moved aside or restored with the right key first. No bytes are written. A missing path, or a file that verifies, is saved as usual. Nothing is renamed or deleted.
- There is no default key in source, and no fallback key.
- The HMAC key is derived with HKDF-SHA256. The info label is `shard-composer-hmac-v1`. That label separates this file format. It is not a key.
- A file signed under one base is rejected under a different base. Rejection means the signed fields are not loaded. The file on disk is left as it was.

### Security Notes

- Changing this value invalidates previously signed state files. Load returns a fresh adapter.
- Use a unique value of at least 32 bytes.
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
