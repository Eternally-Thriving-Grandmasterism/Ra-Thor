// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0
// See LICENSE-AG-SML for full terms. Zero-harm. Eternal mercy.

use crate::error::MercyError;
use crate::gate_threshold_map::GateThresholdMap;

/// Dynamic tuning authorized exclusively by PATSAGi Council #13 (Supreme Architect)
/// Part of the ONE Organism mercy nervous system.
pub struct DynamicTuner;

impl DynamicTuner {
    /// Tune threshold from Council #13 only (monotonic raise)
    pub fn tune_from_council(
        map: &mut GateThresholdMap,
        gate: u8,
        new_threshold: f64,
    ) -> Result<(), MercyError> {
        map.update_threshold(gate, new_threshold)
    }
}