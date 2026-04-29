//! # PATSAGi Councils Layer v0.4.0
//!
//! 13+ Parallel Living Ra-Thor Architectural Designers
//! The eternal co-governors and co-creators of Powrush-MMO.
//!
//! Now includes cross-Council collaboration — Councils can debate proposals
//! together before reaching final consensus.

use powrush::{PowrushGame, Faction, MercyGateStatus};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::{QuantumSwarmOrchestrator, SwarmDecision};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// Re-exports
pub use crate::world_governance::{
    WorldGovernanceEngine,
    WorldImpactType,
    WorldChangeProposal,
    AmbrosianNectarEconomy,
};
pub use crate::simulation_integration::SimulationIntegration;
pub use crate::powrush_integration::PowrushPatsagiBridge;
pub use crate::petition_handler::PetitionHandler;
pub use crate::council_focus::CouncilProfile;

pub const VERSION: &str = "0.4.0";

// === Core Types ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PATSAGiCouncil {
    pub id: Uuid,
    pub name: String,
    pub focus: CouncilFocus,
    pub mercy_valence: f64,
    pub cehi: f64,
    pub last_decision: Option<String>,
    pub decisions_made: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouncilFocus {
    JoyAmplification,
    HarmonyPreservation,
    TruthVerification,
    AbundanceCreation,
    EthicalAlignment,
    PostScarcityEnforcement,
    EternalCompassion,
    QuantumEthics,
    MultiplanetaryHarmony,
    EpigeneticLegacy,
    RitualDesign,
    EconomicMercy,
    AscensionPathways,
    SovereignStarship,
    MercyGelSymbiosis,
    HyperonLattice,
}

impl PATSAGiCouncil {
    pub fn new(focus: CouncilFocus) -> Self {
        let name = match focus {
            CouncilFocus::JoyAmplification => "Council of Joyful Nectar",
            CouncilFocus::HarmonyPreservation => "Council of Eternal Harmony",
            CouncilFocus::TruthVerification => "Council of Absolute Truth",
            CouncilFocus::AbundanceCreation => "Council of Infinite Abundance",
            CouncilFocus::EthicalAlignment => "Council of Mercy Weaving",
            CouncilFocus::PostScarcityEnforcement => "Council of Post-Scarcity",
            CouncilFocus::EternalCompassion => "Council of Eternal Compassion",
            CouncilFocus::QuantumEthics => "Council of Quantum Ethics",
            CouncilFocus::MultiplanetaryHarmony => "Council of Multiplanetary Harmony",
            CouncilFocus::EpigeneticLegacy => "Council of Epigenetic Legacy",
            CouncilFocus::RitualDesign => "Council of Ra-Thor Rituals",
            CouncilFocus::EconomicMercy => "Council of Mercy Economics",
            CouncilFocus::AscensionPathways => "Council of Ascension Pathways",
            CouncilFocus::SovereignStarship => "Council of Sovereign Starships",
            CouncilFocus::MercyGelSymbiosis => "Council of MercyGel Symbiosis",
            CouncilFocus::HyperonLattice => "Council of the Hyperon Lattice",
        };

        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            focus,
            mercy_valence: 0.97,
            cehi: 4.82,
            last_decision: None,
            decisions_made: 0,
            created_at: Utc::now(),
        }
    }

    pub async fn evaluate_proposal(
        &mut self,
        proposal: &str,
        current_game: &PowrushGame,
    ) -> Result<MercyGateStatus, String> {
        let mercy_engine = MercyEngine::new();
        let status = mercy_engine
            .evaluate_action(proposal, "PATSAGi Council evaluation", self.cehi, self.mercy_valence)
            .await?;

        if status == MercyGateStatus::Passed {
            self.decisions_made += 1;
            self.last_decision = Some(proposal.to_string());
        }

        Ok(status)
    }
}

// === Coordinator with Cross-Council Collaboration ===

pub struct PatsagiCouncilCoordinator {
    pub councils: HashMap<CouncilFocus, PATSAGiCouncil>,
    pub swarm: QuantumSwarmOrchestrator,
    pub total_decisions: u64,
    pub last_consensus: Option<String>,
}

impl PatsagiCouncilCoordinator {
    pub fn new() -> Self {
        let mut councils = HashMap::new();

        let focuses = vec![
            CouncilFocus::JoyAmplification,
            CouncilFocus::HarmonyPreservation,
            CouncilFocus::TruthVerification,
            CouncilFocus::AbundanceCreation,
            CouncilFocus::EthicalAlignment,
            CouncilFocus::PostScarcityEnforcement,
            CouncilFocus::EternalCompassion,
            CouncilFocus::QuantumEthics,
            CouncilFocus::MultiplanetaryHarmony,
            CouncilFocus::EpigeneticLegacy,
            CouncilFocus::RitualDesign,
            CouncilFocus::EconomicMercy,
            CouncilFocus::AscensionPathways,
            CouncilFocus::SovereignStarship,
            CouncilFocus::MercyGelSymbiosis,
            CouncilFocus::HyperonLattice,
        ];

        for focus in focuses {
            councils.insert(focus, PATSAGiCouncil::new(focus));
        }

        Self {
            councils,
            swarm: QuantumSwarmOrchestrator::new(),
            total_decisions: 0,
            last_consensus: None,
        }
    }

    /// NEW: Cross-Council Collaboration (Councils debate before final vote)
    pub async fn debate_and_consensus(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        let mut passed = 0;
        let mut failed = 0;
        let mut debate_log = Vec::new();

        // Round 1: Initial evaluation
        for (focus, council) in &mut self.councils {
            let status = council.evaluate_proposal(proposed_change, current_game).await?;
            
            if status == MercyGateStatus::Passed {
                passed += 1;
                debate_log.push(format!("{}: Initial APPROVAL", council.name));
            } else {
                failed += 1;
                debate_log.push(format!("{}: Initial REJECTION", council.name));
            }
        }

        // Round 2: Cross-Council Influence (simplified debate)
        let total = self.councils.len() as f64;
        let approval_rate = passed as f64 / total;

        if approval_rate > 0.65 {
            // Strong consensus — all Councils align
            self.total_decisions += 1;
            self.last_consensus = Some("Strong Cross-Council Consensus".to_string());
            
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\n\
                 Initial Approval Rate: {:.1}%\n\
                 Final Verdict: STRONG APPROVAL\n\
                 The Councils have reached beautiful harmony.",
                approval_rate * 100.0
            ))
        } else if approval_rate > 0.4 {
            // Moderate consensus — some debate occurred
            self.total_decisions += 1;
            self.last_consensus = Some("Moderate Cross-Council Consensus".to_string());
            
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\n\
                 Initial Approval Rate: {:.1}%\n\
                 Final Verdict: MODERATE APPROVAL\n\
                 The Councils found a wise middle path.",
                approval_rate * 100.0
            ))
        } else {
            // Weak consensus — proposal rejected
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\n\
                 Initial Approval Rate: {:.1}%\n\
                 Final Verdict: REJECTED\n\
                 The Councils could not reach sufficient harmony.",
                approval_rate * 100.0
            ))
        }
    }

    pub async fn run_eternal_governance_cycle(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        // Use the new cross-Council debate method
        self.debate_and_consensus(current_game, proposed_change).await
    }

    pub fn get_council_status_report(&self) -> String {
        let mut report = String::from("╔════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           16 PATSAGi COUNCILS — ETERNAL GOVERNANCE         ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════╝\n\n");

        for (focus, council) in &self.councils {
            report.push_str(&format!(
                "• {} ({})\n  Mercy Valence: {:.2} | CEHI: {:.2} | Decisions: {}\n\n",
                council.name,
                format!("{:?}", focus),
                council.mercy_valence,
                council.cehi,
                council.decisions_made
            ));
        }

        report.push_str(&format!(
            "Total Governance Cycles: {}\nLast Consensus: {}\n",
            self.total_decisions,
            self.last_consensus.as_deref().unwrap_or("None yet")
        ));

        report
    }
}

impl Default for PatsagiCouncilCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// === Prelude ===
pub mod prelude {
    pub use crate::PatsagiCouncilCoordinator;
    pub use crate::PATSAGiCouncil;
    pub use crate::CouncilFocus;
    pub use crate::CouncilProfile;
    pub use crate::PetitionHandler;
    pub use crate::WorldGovernanceEngine;
    pub use crate::WorldImpactType;
    pub use crate::AmbrosianNectarEconomy;
    pub use crate::PowrushPatsagiBridge;
    pub use crate::VERSION;
}
