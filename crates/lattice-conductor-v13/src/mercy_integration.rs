// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

//! Mercy Integration Layer for Lattice Conductor v13.1
//! The living mercy nervous system is now wired into the central conductor.

use mercy_gating_runtime::{MercyGatingRuntime, MercyError};
use std::collections::HashMap;

/// ONE Organism mercy-gated evaluation entry point
pub fn evaluate_with_full_mercy(
    runtime: &MercyGatingRuntime,
    scores: &HashMap<u8, f64>,
) -> Result<(), MercyError> {
    runtime.evaluate(scores)
}

/// PATSAGi Council #13 dynamic tuning hook
pub fn council_13_authorized_tune(
    runtime: &mut MercyGatingRuntime,
    gate: u8,
    new_threshold: f64,
) -> Result<(), MercyError> {
    runtime.apply_council_tuning(gate, new_threshold)
}