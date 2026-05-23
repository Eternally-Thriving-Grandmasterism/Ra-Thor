// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0
// See LICENSE-AG-SML for full terms. Zero-harm. Eternal mercy.

use crate::error::MercyError;
use crate::gate_threshold_map::GateThresholdMap;

/// Hot-reload manager with strict monotonic soundness (Lean-corresponding)
/// Ensures ONE Organism mercy nervous system updates remain safe and monotonic.
pub struct HotReloadManager;

impl HotReloadManager {
    pub fn perform_hot_reload(
        current: &mut GateThresholdMap,
        proposed: GateThresholdMap,
    ) -> Result<(), MercyError> {
        for (gate, &new_val) in &proposed.thresholds {
            if let Some(&old_val) = current.get(*gate) {
                if new_val < old_val {
                    return Err(MercyError::HotReloadSoundnessFailed {
                        reason: format!(
                            "Gate {} would decrease from {} to {} — monotonicity violation",
                            gate, old_val, new_val
                        ),
                    });
                }
            }
        }
        *current = proposed;
        Ok(())
    }
}