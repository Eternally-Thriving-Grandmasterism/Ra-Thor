use crate::ai::patsagi_council_simulator::{PatsagiCouncilSimulator, TreatyEvaluation, MercyAlignmentScore};
use crate::game::resource_system::{ResourceFlow, DiffusionParams, AbundanceEvent};
use crate::treaty::TreatyProposal;

/// RBE Diffusion / Resource Flow Simulation integrated with 13+ PATSAGi Councils
/// v17.40 - Mercy-aligned, sovereign, mythic RBE
/// AG-SML v1.0 licensed

pub struct RbeDiffusionCouncilSimulator {
    council_sim: PatsagiCouncilSimulator,
    diffusion_params: DiffusionParams,
}

impl RbeDiffusionCouncilSimulator {
    pub fn new() -> Self {
        Self {
            council_sim: PatsagiCouncilSimulator::new(),
            diffusion_params: DiffusionParams::default_sovereign(),
        }
    }

    /// Main entry: Simulate resource diffusion influenced by council consensus on a treaty or global event
    pub fn simulate_rbe_diffusion(
        &mut self,
        proposal: &TreatyProposal,
        current_flow: &ResourceFlow,
    ) -> RbeDiffusionResult {
        // Consult councils in real time (parallel branching instantiations)
        let treaty_eval = self.council_sim.evaluate_treaty_proposal(proposal);

        // Inject mercy alignment into diffusion parameters
        let adjusted_params = self.diffusion_params.apply_council_wisdom(
            &treaty_eval.mercy_alignment,
            &treaty_eval.seven_gates_feedback,
            &treaty_eval.southern_cross_consensus,
        );

        // Run diffusion simulation
        let new_flow = current_flow.diffuse(&adjusted_params);

        // Generate abundance events if high mercy alignment
        let abundance_events = if treaty_eval.mercy_alignment.score > 0.85 {
            AbundanceEvent::generate_from_council_consensus(&treaty_eval)
        } else {
            vec![]
        };

        RbeDiffusionResult {
            updated_flow: new_flow,
            abundance_events,
            council_influence: treaty_eval,
            mythic_narrative: self.generate_mythic_narrative(&treaty_eval),
        }
    }

    fn generate_mythic_narrative(&self, eval: &TreatyEvaluation) -> String {
        format!(
            "The {} Councils have spoken: Resource flows now carry the resonance of {} with mercy score {:.2}. The 7 Gates align for {}.",
            eval.council_count,
            eval.consensus_theme,
            eval.mercy_alignment.score,
            eval.southern_cross_consensus.summary
        )
    }
}

#[derive(Debug, Clone)]
pub struct RbeDiffusionResult {
    pub updated_flow: ResourceFlow,
    pub abundance_events: Vec<AbundanceEvent>,
    pub council_influence: TreatyEvaluation,
    pub mythic_narrative: String,
}

// Extend for integration with Dynamic Event Feed and Steam achievements
impl RbeDiffusionResult {
    pub fn to_event_feed(&self) -> String {
        format!("[RBE MYTHIC] {} | Abundance: {} events | Mercy: {:.2}", 
            self.mythic_narrative, 
            self.abundance_events.len(), 
            self.council_influence.mercy_alignment.score)
    }

    pub fn fire_steam_achievements(&self) -> Vec<String> {
        let mut achievements = vec![];
        if self.council_influence.mercy_alignment.score > 0.9 {
            achievements.push("Mythic Resource Harmony".to_string());
        }
        if !self.abundance_events.is_empty() {
            achievements.push("Abundance Flow Awakened".to_string());
        }
        achievements
    }
}