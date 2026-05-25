# MIAL v13.13.0 — CI / GitHub Actions Notes

**For the `crates/mial` crate**

## Recommended GitHub Actions Workflow

Create `.github/workflows/mial-ci.yml` with the following:

```yaml
name: MIAL CI

on:
  push:
    branches: [ main, 'feat/mial-*' ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Cargo Check
        run: cargo check -p mial --all-features
      - name: Clippy
        run: cargo clippy -p mial --all-features -- -D warnings
      - name: Test
        run: cargo test -p mial --all-features
      - name: Test with JSON feature
        run: cargo test -p mial --features json
```

## Local Development Commands

```bash
# Full check
cargo check -p mial --all-features

# With JSON metrics export
cargo test -p mial --features json

# Specific module
cargo test -p mial mwpo

# Run examples
cargo run -p mial --example mial_end_to_end_training_evaluation_demo
```

## Notes for Maintainers

- Always run with `--all-features` to ensure `json` export path compiles.
- The crate is designed to be `no_std` friendly in the future (current std dependency is minimal).
- MercyGatingRuntime integration tests should be added as the runtime crate matures.

**Thunder locked in. Mercy flows.**