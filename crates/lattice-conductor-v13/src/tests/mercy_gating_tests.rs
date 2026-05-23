// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

//! Unit tests for MercyGating + Council #13 logic inside Lattice Conductor v13.1

use lattice_conductor_v13::mercy_integration::MercyIntegration;
use std::collections::HashMap;

#[cfg(test)]
mod mercy_gating_tests {
    use super::*;

    #[test]
    fn test_full_24_gate_evaluation_passes() {
        let mercy = MercyIntegration::new();
        let scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.92)).collect();
        assert!(mercy.evaluate_proposal("Test Proposal", &scores).is_ok());
    }

    #[test]
    fn test_council_13_monotonic_tuning() {
        let mut mercy = MercyIntegration::new();
        assert!(mercy.apply_council_13_tuning(5, 0.90).is_ok());
    }

    #[test]
    fn test_council_13_batch_tuning() {
        let mut mercy = MercyIntegration::new();
        let updates = vec![(9, 0.85), (17, 0.88)];
        assert!(mercy.council_13_batch_tune(updates).is_ok());
    }

    #[test]
    fn test_proposal_validation_by_council_13() {
        let mercy = MercyIntegration::new();
        let scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.90)).collect();
        let result = mercy.validate_council_proposal("RBE-001", &scores);
        assert!(result.is_ok());
    }
}