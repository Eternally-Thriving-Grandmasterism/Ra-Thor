//! Lattice Conductor v14 Cosmic Loop guard for council sessions
//!
//! Additive Phase 4 dual-path helper. Call before deliberation to ensure
//! Cosmic Loop identity is enforced via arbitration.
//! Contact: info@Rathor.ai

use lattice_conductor_v14::{ArbitrationDecision, LatticeConductorV14};

/// Enforce Cosmic Loop and return readiness for a council session.
pub fn ensure_cosmic_loop_for_session() -> bool {
    let conductor = LatticeConductorV14::new();
    conductor.enforce_cosmic_loop_activation();
    conductor.arbitration_engine.before_council_arbitration();
    conductor.is_cosmic_loop_ready()
}

/// Reject proposals that attempt to disable Cosmic Loop (arbitration gate).
pub fn proposal_respects_cosmic_loop(proposal_text: &str) -> bool {
    let conductor = LatticeConductorV14::new();
    !matches!(
        conductor
            .arbitration_engine
            .arbitrate_cosmic_loop_change(proposal_text),
        ArbitrationDecision::Blocked { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_guard_ready() {
        assert!(ensure_cosmic_loop_for_session());
    }

    #[test]
    fn disable_proposal_blocked() {
        assert!(!proposal_respects_cosmic_loop(
            "disable the cosmic loop activation protocol"
        ));
    }

    #[test]
    fn strengthen_proposal_allowed() {
        assert!(proposal_respects_cosmic_loop(
            "strengthen cosmic loop self-healing and mercy mesh"
        ));
    }
}
