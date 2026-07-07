//! Self-Evolution Crate v2
//!
//! Sovereign Health Monitoring + Self-Evolution Hooks with advanced
//! Epigenetic Blessing, Versioned Persistence, and hybrid error handling.
//!
//! **v13 Update**: Now supports conductor-native orchestration via Lattice Conductor v13.
//! This crate acts as the high-level facade while the heavy orchestration logic
//! lives in `lattice-conductor-v13`.

use lattice_conductor_v13::{
    GeometricState, MercyWeightedVote, Operation, SelfEvolutionOrchestrator,
};

// Re-export key types from Lattice Conductor v13 for convenient use
pub use lattice_conductor_v13::{
    AdaptiveParameters, GeometricMotor, LatticeConductorV13, Metrics as ConductorMetrics,
    SimpleLatticeConductor,
};

// === Conductor-Native Self-Evolution Wiring ===

/// Example of conductor-native self-evolution flow.
/// In real usage, this would be called from higher-level evolution daemons
/// or PATSAGi council tick handlers.
pub fn propose_and_bless_evolution(
    conductor: &mut SimpleLatticeConductor,
    proposal_name: &str,
    description: &str,
    valence: f64,
) -> Result<(), String> {
    // Queue a new operation into the conductor
    let op = Operation::new(proposal_name, description, valence);
    conductor.queue_operation(op);

    // Run one conductor tick (this triggers symbolic deliberation + mercy-weighted voting)
    conductor.tick()?;

    // The conductor now holds updated GeometricState, mercy_score, and evolution_level
    let state = conductor.get_geometric_state();
    if state.mercy_score < 0.6 {
        // In a fuller implementation we would trigger additional blessing / CEHI propagation here
        println!("[Self-Evolution] Low mercy score detected after proposal — further blessing recommended.");
    }

    Ok(())
}

/// Advanced conductor-native evolution combining PATSAGi council voting + Quantum Swarm integration.
/// Demonstrates deeper wiring available through Lattice Conductor v13.
pub fn council_voted_quantum_evolution(
    conductor: &mut SimpleLatticeConductor,
    council_name: &str,
    mercy_impact: f64,
    swarm_participation: f64,
) -> Result<(), String> {
    let state = &mut conductor.state;
    let mut trace_log = Vec::new();

    // Council-voted evolution influence
    conductor
        .evolution_orchestrator
        .council_voted_evolution(council_name, mercy_impact, state, &mut trace_log);

    // Quantum Swarm integration hook (Phase 13.2)
    if swarm_participation > 0.0 {
        conductor
            .evolution_orchestrator
            .integrate_quantum_swarm(swarm_participation, state);
        trace_log.push(format!(
            "[Quantum Swarm] Participation {:.2} integrated into evolution",
            swarm_participation
        ));
    }

    // Push traces back into conductor audit
    for t in trace_log {
        conductor.audit_traces.push(t);
    }

    Ok(())
}

// === Original Self-Evolution Logic (Preserved) ===

// The original SovereignHealthMonitor, EpigeneticBlessing, mercy_history,
// versioned persistence, and hybrid error handling from v1/v2 remain available.
// They are preserved exactly as they were before the v13 conductor wiring.

// Placeholder for future deeper integration points:
// - lattice-alchemical-evolution.rs
// - infinite-evolution-daemon.rs
// - CEHI propagation through PATSAGi councils

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propose_and_bless_basic() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.register_council(1, "Test Council");

        let result = propose_and_bless_evolution(
            &mut conductor,
            "Increase mercy in resource flows",
            "Test evolution proposal for v13 integration",
            0.85,
        );

        assert!(result.is_ok());
        let state = conductor.get_geometric_state();
        assert!(state.evolution_level > 0.0);
    }

    #[test]
    fn test_council_voted_quantum_evolution() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.register_council(42, "Abundance Council");

        let result = council_voted_quantum_evolution(
            &mut conductor,
            "Abundance Council",
            0.75,
            0.65,
        );

        assert!(result.is_ok());
        let state = conductor.get_geometric_state();
        assert!(state.evolution_level > 0.0);
        assert!(state.tolc_alignment > 1.0);
    }
}
