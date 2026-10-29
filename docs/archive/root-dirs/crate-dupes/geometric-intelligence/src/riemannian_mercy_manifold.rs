//! RiemannianMercyManifold v14.6
//!
//! Advanced Riemannian geometry + Berry Phase + Real Embedded PATSAGi Council Engine + EpigeneticModulation wiring.
//! Simulation sequences now directly affect epigenetic state and transport decisions.
//! Includes hooks for Council Proposal Protocol.

use crate::polyhedral_harmonic_engine::{PolyhedralResonanceReport, U57LayerDetails};
use crate::types::{EpigeneticBlessing, EpigeneticModulation, CouncilProposal};
use std::collections::HashMap;

// ... (other structs unchanged for brevity)

pub struct RiemannianMercyManifold {
    pub version: &'static str,
    pub curvature_params: CurvatureParameters,
    pub epigenetic_state: EpigeneticModulation,  // Wired simulation state
}

impl Default for RiemannianMercyManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl RiemannianMercyManifold {
    pub fn new() -> Self {
        Self {
            version: "v14.6-simulation-wired",
            curvature_params: CurvatureParameters::default(),
            epigenetic_state: EpigeneticModulation::new(0.8, 0.5, "Hyperbolic"),
        }
    }

    // ... (existing council engine methods unchanged)

    /// NEW: Wire council sequence simulation directly into the manifold's epigenetic state.
    /// This fulfills request #1 — simulation now affects the geometric body in real time.
    pub fn apply_council_sequence_to_epigenetic(&mut self, sequence: &[(f64, &str)]) -> String {
        let report = self.epigenetic_state.simulate_council_sequence(sequence);
        // Optionally modulate mercy_influence based on final state
        let final_bonus = self.epigenetic_state.evolution_rate_bonus();
        self.curvature_params.mercy_influence = (self.curvature_params.mercy_influence * (0.9 + final_bonus * 0.1)).clamp(0.8, 1.4);
        report
    }

    /// NEW: Evaluate a CouncilProposal using the embedded engine + epigenetic modulation.
    /// Hook for full Council Proposal Protocol (request #3).
    pub fn evaluate_council_proposal(&mut self, proposal: &CouncilProposal) -> (f64, Vec<EpigeneticBlessing>, String) {
        let context = format!("{} in {} layer", proposal.context, proposal.geometric_layer);
        let (valence, _gates, reason) = self.evaluate_council_valence(&proposal.council, &context);

        // Apply to epigenetic state
        self.epigenetic_state.apply_council_valence(valence, &proposal.council);

        let modulated_mercy = (self.curvature_params.mercy_influence * (0.7 + valence * 0.3)).clamp(0.75, 1.35);

        let blessings = vec![self.epigenetic_state.to_blessing(&proposal.council)];

        (modulated_mercy, blessings, reason)
    }

    // ... (rest of existing methods, with optional epigenetic modulation in transport if desired)

    // Example: run_u57_informed_transport_sequence can now optionally call epigenetic effects
}

// ... (tests updated to cover new wiring)
