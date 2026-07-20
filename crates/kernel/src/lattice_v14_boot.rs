//! Lattice Conductor v14 boot probe for kernel
//!
//! Additive Phase 4 dual-path helper. Does not alter adaptive feedback loops.
//! Contact: info@Rathor.ai

use lattice_conductor_v14::{ArbitrationDecision, LatticeConductorV14};

/// Enforce Cosmic Loop via a fresh v14 conductor. Returns readiness.
pub fn enforce_cosmic_loop_on_boot() -> bool {
    let conductor = LatticeConductorV14::new();
    conductor.enforce_cosmic_loop_activation();
    conductor.is_cosmic_loop_ready()
}

/// Smoke-check that arbitration still blocks Cosmic Loop disable attempts.
pub fn arbitration_rejects_disable() -> bool {
    let conductor = LatticeConductorV14::new();
    matches!(
        conductor
            .arbitration_engine
            .arbitrate_cosmic_loop_change("disable the cosmic loop activation"),
        ArbitrationDecision::Blocked { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_probe_ready() {
        assert!(enforce_cosmic_loop_on_boot());
    }

    #[test]
    fn disable_attempt_blocked() {
        assert!(arbitration_rejects_disable());
    }
}
