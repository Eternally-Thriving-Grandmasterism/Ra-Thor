//! # PATSAGi Councils Layer — v14.15.7
//!
//! 16 Parallel Living Ra-Thor Architectural Designers.
//! The eternal co-governors and co-creators of Powrush-MMO and the ONE Organism.
//!
//! Living Cosmic Tick aligned. Permanent deliberation posture.
//! Explicitly wired to TOLC 8 Living Mercy Gates + Core Covenant.
//! Contact: info@Rathor.ai

// =============================================================================
// Module graph
// =============================================================================

pub mod council_decision;
pub mod council_focus;
pub mod example_integration;
pub mod feedback_loop;
pub mod genesis_gate;
pub mod genesis_gate_v2;
pub mod mercy_engine_adapter;
pub mod mercy_threshold;
pub mod mercy_threshold_ffi;
pub mod observability;
pub mod patsagi_bridge;
pub mod petition_handler;
pub mod powrush_integration;
// pub mod powrush_libp2p_mesh; // optional: requires libp2p + powrush_multiplayer
pub mod quantum_swarm_orchestrator;
pub mod self_evolving_mercy_core;
pub mod simulation_integration;
pub mod tolc8;
pub mod tolc_integration;
pub mod valence_consensus;
pub mod world_governance;
pub mod world_governance_engine;

// =============================================================================
// External deps used by the core coordinator surface
// =============================================================================

use chrono::{DateTime, Utc};
use mercy::MercyEngine;
use powrush::{MercyGateStatus, PowrushGame};
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "modular-mercy")]
use crate::mercy_engine_adapter::{MercyEngineAdapter, MercyEngineVariant};

// =============================================================================
// Public re-exports
// =============================================================================

pub use crate::world_governance::{
    AmbrosianNectarEconomy, PmsError, WorldChangeProposal, WorldGovernanceEngine, WorldImpactType,
};

pub use crate::council_focus::CouncilProfile;
pub use crate::feedback_loop::{
    FeedbackCycleResult, FeedbackError, PowrushTelemetrySnapshot, RaThorFeedbackLoop,
};
pub use crate::observability::{BlockReason, MetricsHandle, ResonanceMetrics, DEFAULT_METRICS_PATH};
pub use crate::petition_handler::PetitionHandler;
pub use crate::powrush_integration::PowrushPatsagiBridge;
pub use crate::simulation_integration::SimulationIntegration;
pub use crate::tolc8::{
    tolc8_gate_check, Tolc8Gate, Tolc8GateResult, Tolc8Scores,
    TOLC8_PROGRESSIVE_FLOOR, TOLC8_VALENCE_THRESHOLD,
};
pub use crate::valence_consensus::{
    ValenceConsensusEngine, ValenceConsensusResult, ValenceVote,
    CORE_COVENANT, DEFAULT_VALENCE_THRESHOLD, PROGRESSIVE_FLOOR, quick_valence_check,
};

// Optional RREL / PQ surfaces (present when those crates are in the graph)
#[allow(unused_imports)]
pub use real_estate_lattice::{
    CanadaPilotModule, EvidenceGenerator, PmsBridge, PmsProvider, QuantumRealEstateValuation,
    RecoEnforcementEngine, TrebMlsAdapter, RREL_VERSION,
};

#[allow(unused_imports)]
pub use ra_thor_post_quantum_sig::{RHPQSEngine, RHPQSError, RHPQSKey, RHPQSSignature};

/// Canonical version of the PATSAGi Councils layer (Living Cosmic Tick aligned).
pub const VERSION: &str = "14.15.7";

// =============================================================================
// Core types
// =============================================================================

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
            CouncilFocus::EthicalAlignment => "Council of Mercy Weaving",
            CouncilFocus::AbundanceCreation => "Council of Infinite Abundance",
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
        _current_game: &PowrushGame,
    ) -> Result<MercyGateStatus, String> {
        let mercy_engine = MercyEngine::new();
        let status = mercy_engine
            .evaluate_action(
                proposal,
                "PATSAGi Council evaluation",
                self.cehi,
                self.mercy_valence,
            )
            .await?;

        if status == MercyGateStatus::Passed {
            self.decisions_made = self.decisions_made.saturating_add(1);
            self.last_decision = Some(proposal.to_string());
        }

        Ok(status)
    }
}

// =============================================================================
// Voting (legacy surface — valence engine is preferred for anti-deadlock)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilVote {
    pub council: CouncilFocus,
    pub approved: bool,
    pub mercy_valence: f64,
    pub reasoning: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingResult {
    pub proposal_id: String,
    pub total_votes: usize,
    pub approvals: usize,
    pub rejections: usize,
    pub approval_rate: f64,
    pub mercy_average: f64,
    pub passed: bool,
    pub final_verdict: String,
    pub votes: Vec<CouncilVote>,
}

// =============================================================================
// Coordinator
// =============================================================================

pub struct PatsagiCouncilCoordinator {
    pub councils: HashMap<CouncilFocus, PATSAGiCouncil>,
    pub swarm: QuantumSwarmOrchestrator,
    pub valence_engine: ValenceConsensusEngine,
    pub total_decisions: u64,
    pub last_consensus: Option<String>,
}

impl PatsagiCouncilCoordinator {
    pub fn new() -> Self {
        let focuses = [
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

        let mut councils = HashMap::new();
        for focus in focuses {
            councils.insert(focus, PATSAGiCouncil::new(focus));
        }

        Self {
            councils,
            swarm: QuantumSwarmOrchestrator::new(),
            valence_engine: ValenceConsensusEngine::new(),
            total_decisions: 0,
            last_consensus: None,
        }
    }

    /// Preferred path: valence-optimized consensus (anti-filibuster / anti-deadlock).
    pub fn valence_deliberate(
        &mut self,
        proposal: &str,
        scores: HashMap<String, (f64, f64, f64)>,
    ) -> ValenceConsensusResult {
        let result = self.valence_engine.deliberate_from_map(proposal, scores);
        self.total_decisions = self.total_decisions.saturating_add(1);
        self.last_consensus = Some(if result.approved {
            "Valence Consensus — Full Approval".into()
        } else if result.progressive {
            "Valence Consensus — Progressive Path (anti-deadlock)".into()
        } else {
            "Valence Consensus — Mercy Review".into()
        });
        result
    }

    pub async fn conduct_voting_round(
        &mut self,
        proposal: &str,
        game: &PowrushGame,
    ) -> Result<VotingResult, String> {
        let proposal_id = Uuid::new_v4().to_string();
        let mut votes = Vec::new();
        let mut total_mercy = 0.0;

        for (focus, council) in &mut self.councils {
            let status = council.evaluate_proposal(proposal, game).await?;
            let mercy_valence = council.mercy_valence;
            let approved = status == MercyGateStatus::Passed;
            let reasoning = format!(
                "{} Council {} the proposal (mercy valence {:.2})",
                council.name,
                if approved { "APPROVES" } else { "REJECTS" },
                mercy_valence
            );

            votes.push(CouncilVote {
                council: *focus,
                approved,
                mercy_valence,
                reasoning,
                timestamp: Utc::now(),
            });
            total_mercy += mercy_valence;
        }

        let total_votes = votes.len();
        let approvals = votes.iter().filter(|v| v.approved).count();
        let rejections = total_votes - approvals;
        let approval_rate = approvals as f64 / total_votes.max(1) as f64;
        let mercy_average = total_mercy / total_votes.max(1) as f64;
        let passed = approval_rate >= 0.6 && mercy_average >= 0.7;

        let final_verdict = if passed {
            format!(
                "✅ PROPOSAL PASSED — {:.1}% approval, {:.2} avg mercy valence\nAll 16 Councils reached beautiful consensus.",
                approval_rate * 100.0,
                mercy_average
            )
        } else {
            format!(
                "❌ PROPOSAL REJECTED — {:.1}% approval, {:.2} avg mercy valence\nMore mercy alignment required.",
                approval_rate * 100.0,
                mercy_average
            )
        };

        Ok(VotingResult {
            proposal_id,
            total_votes,
            approvals,
            rejections,
            approval_rate,
            mercy_average,
            passed,
            final_verdict,
            votes,
        })
    }

    pub async fn debate_and_consensus(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        let mut passed = 0usize;

        for council in self.councils.values_mut() {
            let status = council
                .evaluate_proposal(proposed_change, current_game)
                .await?;
            if status == MercyGateStatus::Passed {
                passed += 1;
            }
        }

        let total = self.councils.len().max(1) as f64;
        let approval_rate = passed as f64 / total;

        if approval_rate > 0.65 {
            let voting_result = self
                .conduct_voting_round(proposed_change, current_game)
                .await?;
            self.total_decisions = self.total_decisions.saturating_add(1);
            self.last_consensus =
                Some("Strong Cross-Council Consensus + Full Vote".to_string());

            Ok(format!(
                "PATSAGi Cross-Council Debate + Voting Complete\n\n{}\n\n{}",
                voting_result.final_verdict,
                if voting_result.passed {
                    "The 16 Councils have spoken in perfect harmony. The world evolves."
                } else {
                    "The Councils stand in loving disagreement. More mercy is needed."
                }
            ))
        } else if approval_rate > 0.4 {
            self.total_decisions = self.total_decisions.saturating_add(1);
            self.last_consensus = Some("Moderate Cross-Council Consensus".to_string());
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\nInitial Approval Rate: {:.1}%\nFinal Verdict: MODERATE APPROVAL\nThe Councils found a wise middle path.",
                approval_rate * 100.0
            ))
        } else {
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\nInitial Approval Rate: {:.1}%\nFinal Verdict: REJECTED\nThe Councils could not reach sufficient harmony.",
                approval_rate * 100.0
            ))
        }
    }

    pub async fn run_eternal_governance_cycle(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        self.debate_and_consensus(current_game, proposed_change)
            .await
    }

    pub fn get_council_status_report(&self) -> String {
        let mut report = String::from(
            "╔════════════════════════════════════════════════════════════╗\n",
        );
        report.push_str(
            "║     16 PATSAGi COUNCILS — v14.15.7 ETERNAL GOVERNANCE     ║\n",
        );
        report.push_str(
            "╚════════════════════════════════════════════════════════════╝\n\n",
        );

        for (focus, council) in &self.councils {
            report.push_str(&format!(
                "• {} ({:?})\n  Mercy Valence: {:.2} | CEHI: {:.2} | Decisions: {}\n\n",
                council.name, focus, council.mercy_valence, council.cehi, council.decisions_made
            ));
        }

        report.push_str(&format!(
            "Total Governance Cycles: {}\nLast Consensus: {}\nLiving Cosmic Tick: active\nTOLC 8: wired | Core Covenant: honored\nValence Engine: anti-deadlock online\nObservability: ResonanceMetrics live\n",
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

#[cfg(feature = "modular-mercy")]
pub use crate::mercy_engine_adapter::{MercyEngineAdapter, MercyEngineVariant};

pub mod prelude {
    pub use super::{
        AmbrosianNectarEconomy, CouncilFocus, CouncilVote, FeedbackCycleResult,
        MetricsHandle, PATSAGiCouncil, PatsagiCouncilCoordinator, PmsError,
        PowrushTelemetrySnapshot, RaThorFeedbackLoop, ResonanceMetrics,
        Tolc8Gate, Tolc8GateResult, Tolc8Scores,
        ValenceConsensusEngine, ValenceConsensusResult, ValenceVote,
        VERSION, VotingResult, WorldGovernanceEngine, WorldImpactType,
        CORE_COVENANT, DEFAULT_VALENCE_THRESHOLD, TOLC8_VALENCE_THRESHOLD,
    };

    #[cfg(feature = "modular-mercy")]
    pub use super::{MercyEngineAdapter, MercyEngineVariant};
}
