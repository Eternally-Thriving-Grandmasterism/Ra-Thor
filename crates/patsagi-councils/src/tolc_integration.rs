//! TOLC Integration Bridge for PATSAGi Councils
//!
//! Allows the 13+ Parallel Councils to consult the TOLC Lattice when making
//! governance decisions, petitions, and world impact actions.

use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;
use crate::WorldGovernanceEngine;

pub struct TOLCCouncilBridge {
    lattice_engine: TOLCLatticeActivationEngine,
    mercy_engine: MercyEngine,
}

impl TOLCCouncilBridge {
    pub fn new() -> Self {
        Self {
            lattice_engine: TOLCLatticeActivationEngine::new(),
            mercy_engine: MercyEngine::new(),
        }
    }

    /// Let councils consult the TOLC lattice before making important decisions
    pub async fn consult_tolc_before_decision(
        &mut self,
        decision_context: &str,
        world: &mut WorldGovernanceEngine,
    ) -> String {
        let mercy_valence = self.mercy_engine.compute_valence(decision_context).await;

        let lattice_insight = self.lattice_engine
            .activate_full_lattice_up_to(35, &mut world.powrush_game)
            .await;

        format!(
            "TOLC Council Consultation Complete\n\nMercy Valence: {:.3}\n\nLattice Insight: {}",
            mercy_valence, lattice_insight
        )
    }

    /// Apply TOLC self-evolution effects after a council decision
    pub fn apply_post_decision_evolution(&mut self, world: &mut WorldGovernanceEngine) -> String {
        self.lattice_engine.quick_eternal_self_evolution_pulse(&mut world.powrush_game)
    }
}
