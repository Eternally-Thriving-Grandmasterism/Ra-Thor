#!/bin/bash

# ================================================
# Ra-Thor Regression Test Runner
# ================================================
# Runs the core regression tests for formal argumentation semantics.
# This helps protect valuable logic from accidental regression.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color


echo -e "${BLUE}=== Ra-Thor Regression Test Runner ===${NC}"

echo -e "\n${YELLOW}[1/2] Running regression_tests module...${NC}"
cargo test --package lattice-conductor-v14 regression_tests -- --nocapture

echo -e "\n${YELLOW}[2/2] Running full test suite for lattice-conductor-v14...${NC}"
cargo test --package lattice-conductor-v14

echo -e "\n${GREEN}=== All regression tests completed successfully ===${NC}"

echo -e "\nTip: You can also run just the regression tests with:\n  cargo test --package lattice-conductor-v14 regression_tests"
