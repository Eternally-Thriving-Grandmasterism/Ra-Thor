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

### `full`
Complete ONE Organism experience.

### `focused-real-estate` (alias: `real-estate`)
Focused on Real Estate workflows.

### `focused-geometry` (alias: `geometry`)
Focused on Sacred Geometry.

## Usage

```bash
cargo xtask build-shard --profile focused-real-estate
```
