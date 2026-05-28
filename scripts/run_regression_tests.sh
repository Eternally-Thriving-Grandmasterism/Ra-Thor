#!/bin/bash

# Ra-Thor Regression Test Runner
# Runs the core regression tests for formal argumentation semantics

set -e

echo "=== Running Ra-Thor Regression Tests ==="

echo "\n[1/2] Running regression_tests module..."
cargo test --package lattice-conductor-v14 regression_tests -- --nocapture

echo "\n[2/2] Running all tests in lattice-conductor-v14..."
cargo test --package lattice-conductor-v14

echo "\n=== All regression tests completed successfully ==="
