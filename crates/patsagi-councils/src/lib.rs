//! patsagi-councils — older council monolith.
//!
//! `CouncilFocus` names 16 focuses. That list is source. It is not a test result.
//! A living cosmic tick is not demonstrated in this build.
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! This package is not `crates/council`.
//! A mercy-block reject is not flipped to apply by a later vote.
//! Contact: info@Rathor.ai
//!
//! Files that are not modules start with the STUB-AUDIT-1 PATSAGI-COUNCILS-BUILDS-1 design-only header.
//! `lean-sys` is not a dependency. An FFI bridge was not demonstrated.

pub mod feedback_loop;
pub mod observability;
pub mod predictive_support;
pub mod security_support;
pub mod tolc8;
pub mod valence_consensus;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Adapter orchestrator from `ra-thor-quantum-swarm-orchestrator`.
/// It does not implement `Clone`. This crate does not add `Clone`.
pub use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

pub use crate::feedback_loop::{
    FeedbackCycleResult, FeedbackError, PowrushTelemetrySnapshot, RaThorFeedbackLoop,
};
pub use crate::observability::{
    BlockReason, MetricsHandle, ResonanceMetrics, DEFAULT_METRICS_PATH,
};
pub use crate::predictive_support::{
    rank_policies_by_efe, CouncilPolicyHint, PredictiveSignal, PredictiveSupportError,
    MERCY_VALENCE_FLOOR,
};
pub use crate::security_support::{
    apply_security_pressure, SecurityCouncilVerdict, SecurityDomainProfile, SecurityRiskTier,
    SecuritySignal, SecuritySupportError, SecurityThreatClass,
};
pub use crate::tolc8::{
    tolc8_gate_check, Tolc8Gate, Tolc8GateResult, Tolc8Scores, TOLC8_PROGRESSIVE_FLOOR,
    TOLC8_VALENCE_THRESHOLD,
};
pub use crate::valence_consensus::{
    council_vote_after_reject, quick_valence_check, CouncilGateLabel, ValenceConsensusEngine,
    ValenceConsensusResult, ValenceVote, CORE_COVENANT, DEFAULT_VALENCE_THRESHOLD,
    PROGRESSIVE_FLOOR,
};

/// Source string. Not a measured release result.
pub const VERSION: &str = "14.15.11";

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
            // Source literals, not measured values.
            mercy_valence: 0.97,
            cehi: 4.82,
            last_decision: None,
            decisions_made: 0,
            created_at: Utc::now(),
        }
    }

    pub fn apply_predictive_uplift(&mut self, signal: &PredictiveSignal) {
        if signal.is_actionable() {
            let uplift = signal.mercy_uplift();
            self.mercy_valence = (self.mercy_valence + uplift).min(1.0);
        }
    }
}

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

/// Outcome of a security-block deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityDeliberationResult {
    pub signal_source: String,
    pub risk_tier: String,
    pub domain: String,
    pub verdict: String,
    pub councils_touched: usize,
    pub valence_pressure_applied: f64,
    pub recommended_focus: String,
    pub consensus_note: String,
}

pub struct PatsagiCouncilCoordinator {
    pub councils: HashMap<CouncilFocus, PATSAGiCouncil>,
    pub swarm: QuantumSwarmOrchestrator,
    pub valence_engine: ValenceConsensusEngine,
    pub total_decisions: u64,
    pub last_consensus: Option<String>,
    pub security_blocks_reviewed: u64,
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
            security_blocks_reviewed: 0,
        }
    }

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

    pub fn apply_predictive_signal(&mut self, signal: &PredictiveSignal) {
        for council in self.councils.values_mut() {
            council.apply_predictive_uplift(signal);
        }
    }

    /// Apply a white-hat security signal (blocked ingest / high risk).
    pub fn apply_security_signal(
        &mut self,
        signal: &SecuritySignal,
    ) -> Result<usize, SecuritySupportError> {
        let touched = apply_security_pressure(&mut self.councils, signal)?;
        self.security_blocks_reviewed = self.security_blocks_reviewed.saturating_add(1);
        self.total_decisions = self.total_decisions.saturating_add(1);
        self.last_consensus = Some(format!(
            "Security Review — {} tier={} domain={} verdict={} pressure={:.4}",
            signal.source_label,
            signal.risk_tier.as_str(),
            signal.domain.as_str(),
            signal.decide_verdict().as_str(),
            signal.valence_pressure()
        ));
        Ok(touched)
    }

    /// Full deliberation record for a blocked ingestion event from ONE Organism.
    pub fn deliberate_security_block(
        &mut self,
        source_label: &str,
        risk_tier_label: &str,
        risk_score: f64,
        threat_labels: &[String],
        findings_count: usize,
        message: &str,
    ) -> Result<SecurityDeliberationResult, SecuritySupportError> {
        self.deliberate_security_block_with_domain(
            source_label,
            risk_tier_label,
            risk_score,
            threat_labels,
            findings_count,
            message,
            SecurityDomainProfile::StrictWhitehat,
        )
    }

    /// Domain-aware white-hat block deliberation (research / enterprise / education / …).
    #[allow(clippy::too_many_arguments)]
    pub fn deliberate_security_block_with_domain(
        &mut self,
        source_label: &str,
        risk_tier_label: &str,
        risk_score: f64,
        threat_labels: &[String],
        findings_count: usize,
        message: &str,
        domain: SecurityDomainProfile,
    ) -> Result<SecurityDeliberationResult, SecuritySupportError> {
        let tier = SecurityRiskTier::from_str_label(risk_tier_label);
        let signal = SecuritySignal::try_new_with_domain(
            source_label,
            tier,
            risk_score,
            threat_labels,
            findings_count,
            true,
            message,
            domain,
        )?;
        let pressure = signal.valence_pressure();
        let focus = signal.recommended_focus_hint().to_string();
        let verdict = signal.decide_verdict();
        let touched = self.apply_security_signal(&signal)?;

        Ok(SecurityDeliberationResult {
            signal_source: source_label.into(),
            risk_tier: risk_tier_label.into(),
            domain: domain.as_str().into(),
            verdict: verdict.as_str().into(),
            councils_touched: touched,
            valence_pressure_applied: pressure,
            recommended_focus: focus,
            consensus_note: format!(
                "White-hat block deliberated under {}. Verdict: {}. {} councils adjusted. Focus: {}.",
                domain.as_str(),
                verdict.as_str(),
                touched,
                signal.recommended_focus_hint()
            ),
        })
    }

    pub fn get_council_status_report(&self) -> String {
        let mut report =
            String::from("╔════════════════════════════════════════════════════════════╗\n");
        report.push_str("║  PATSAGi Councils (16 focuses, source list) — v14.15.11   ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════╝\n\n");

        for (focus, council) in &self.councils {
            report.push_str(&format!(
                "• {} ({:?})\n  mercy_valence (source literal): {:.2} | cehi (source literal): {:.2} | Decisions: {}\n\n",
                council.name, focus, council.mercy_valence, council.cehi, council.decisions_made
            ));
        }

        report.push_str(&format!(
            "Total Governance Cycles: {}\nSecurity Blocks Reviewed: {}\nLast Consensus: {}\nLiving Cosmic Tick: not demonstrated in this build\nSwarm adapters registered: {}\nSwarm version: {}\nTOLC 8 names: Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony\nValence Engine: anti-deadlock path is the progressive floor in source\nPredictive Support: reception helper\nSecurity Support: white-hat signal helper\nObservability: ResonanceMetrics counters\n",
            self.total_decisions,
            self.security_blocks_reviewed,
            self.last_consensus.as_deref().unwrap_or("None yet"),
            self.swarm.registered_system_count(),
            self.swarm.version
        ));

        report
    }
}

impl Default for PatsagiCouncilCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

pub mod prelude {
    pub use super::{
        council_vote_after_reject, tolc8_gate_check, BlockReason, CouncilFocus, CouncilGateLabel,
        CouncilPolicyHint, CouncilVote, FeedbackCycleResult, FeedbackError, MetricsHandle,
        PATSAGiCouncil, PatsagiCouncilCoordinator, PowrushTelemetrySnapshot, PredictiveSignal,
        PredictiveSupportError, QuantumSwarmOrchestrator, RaThorFeedbackLoop, ResonanceMetrics,
        SecurityCouncilVerdict, SecurityDeliberationResult, SecurityDomainProfile,
        SecurityRiskTier, SecuritySignal, SecuritySupportError, SecurityThreatClass, Tolc8Gate,
        Tolc8GateResult, Tolc8Scores, ValenceConsensusEngine, ValenceConsensusResult, ValenceVote,
        VotingResult, CORE_COVENANT, DEFAULT_METRICS_PATH, DEFAULT_VALENCE_THRESHOLD,
        MERCY_VALENCE_FLOOR, PROGRESSIVE_FLOOR, TOLC8_PROGRESSIVE_FLOOR, TOLC8_VALENCE_THRESHOLD,
        VERSION,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_inserts_sixteen_focuses_and_an_empty_adapter_swarm() {
        let coordinator = PatsagiCouncilCoordinator::new();
        assert_eq!(coordinator.councils.len(), 16);
        assert_eq!(
            coordinator
                .councils
                .get(&CouncilFocus::TruthVerification)
                .map(|council| council.name.as_str()),
            Some("Council of Absolute Truth")
        );
        assert_eq!(
            coordinator
                .councils
                .get(&CouncilFocus::HyperonLattice)
                .map(|council| council.decisions_made),
            Some(0)
        );
        assert_eq!(coordinator.swarm.registered_system_count(), 0);
        assert_eq!(coordinator.swarm.version, "v14.0-one-organism");
        assert_eq!(coordinator.total_decisions, 0);
        assert_eq!(coordinator.security_blocks_reviewed, 0);
        assert_eq!(VERSION, "14.15.11");
        let report = coordinator.get_council_status_report();
        assert!(report.contains("Living Cosmic Tick: not demonstrated in this build"));
        assert!(report.contains("Swarm adapters registered: 0"));
    }
}
