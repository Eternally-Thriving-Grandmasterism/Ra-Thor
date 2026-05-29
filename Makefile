# Ra-Thor Lattice Conductor v14 - Development & Regression Commands

.PHONY: test test-verbose test-quiet regression ci

# Run all tests (recommended for regression)
test:
	cargo test

# Verbose test output
test-verbose:
	cargo test -- --nocapture

# Quiet test run (good for CI)
test-quiet:
	cargo test --quiet

# Full regression suite (core modules)
regression:
	cargo test self_evolution
	cargo test hybrid_sovereign_channel
	cargo test post_quantum_signatures
	cargo test governance

# CI-friendly target
ci: test-quiet

# Future: Add clippy, fmt check, etc.
check:
	cargo check
	cargo clippy -- -D warnings
