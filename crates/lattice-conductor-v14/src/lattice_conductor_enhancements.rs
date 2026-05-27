// crates/lattice-conductor-v14/src/lattice_conductor_enhancements.rs
// Structured Logging for Governance Risk Scores

use crate::distributed_mercy_mesh::{DistributedMercyMesh, MercyEvent, OrganismNode};
use crate::patsagi_governance::{PatsagiCouncilSimulator, PatsagiDecision, PatsagiReviewRequest};
use crate::cooperative_governance::CooperativeGame;
use std::collections::HashSet;

/// Structured report for governance risk analysis
#[derive(Debug, Clone)]
pub struct GovernanceRiskReport {
    pub risk_score: f64,
    pub max_banzhaf: f64,
    pub shapley_variance: f64,
    pub mercy_alignment: f64,
    pub recommended_action: String,
}

impl GovernanceRiskReport {
    pub fn log(&self) {
        println!("[GOVERNANCE RISK] score={:.3} | max_banzhaf={:.3} | shapley_var={:.3} | mercy={:.3} | action={}",
            self.risk_score,
            self.max_banzhaf,
            self.shapley_variance,
            self.mercy_alignment,
            self.recommended_action
        );
    }
}

pub struct LatticeConductorEnhancements;

impl LatticeConductorEnhancements {
    pub fn enforce_one_organism_identity(mesh: &mut DistributedMercyMesh) -> bool {
        if !mesh.verify_unified_core_health() {
            let unified = OrganismNode::new_unified_core();
            mesh.register_organism(unified);
            return mesh.verify_unified_core_health();
        }
        true
    }

    pub fn run_full_lattice_diagnostics(mesh: &DistributedMercyMesh) -> LatticeDiagnosticsReport {
        let unified_healthy = mesh.verify_unified_core_health();
        LatticeDiagnosticsReport {
            unified_organism_healthy: unified_healthy,
            pending_healing_requests: mesh.get_pending_requests().len(),
            total_audit_entries: mesh.get_audit_log().len(),
            average_mercy_alignment: 0.95,
            hybrid_channels_active: true,
            overall_status: if unified_healthy { "Healthy".to_string() } else { "Degraded".to_string() },
        }
    }

    pub fn request_patsagi_review(
        mesh: &DistributedMercyMesh,
        topic: &str,
        summary: &str,
    ) -> PatsagiReviewRequest {
        let report = Self::run_full_lattice_diagnostics(mesh);
        PatsagiReviewRequest {
            topic: topic.to_string(),
            summary: summary.to_string(),
            mercy_impact_score: report.average_mercy_alignment,
            requested_by: "lattice-conductor".to_string(),
        }
    }

    pub fn evaluate_patsagi_coalition(
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
        let game = CooperativeGame::new(participants, coalition_value_fn);
        (game.shapley_value(), game.banzhaf_index())
    }

    pub fn calculate_governance_risk_score(
        max_banzhaf: f64,
        shapley_variance: f64,
        mercy_alignment: f64,
    ) -> f64 {
        let power_risk = max_banzhaf.clamp(0.0, 1.0);
        let fairness_risk = shapley_variance.min(1.0);
        let base_risk = (0.55 * power_risk) + (0.45 * fairness_risk);
        let mercy_factor = 1.0 - (mercy_alignment * 0.15);
        (base_risk * mercy_factor).clamp(0.0, 1.0)
    }

    /// Returns both decision and structured risk report
    pub fn submit_to_patsagi_with_game_theory(
        mesh: &mut DistributedMercyMesh,
        topic: &str,
        summary: &str,
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> (PatsagiDecision, GovernanceRiskReport) {
        let request = Self::request_patsagi_review(mesh, topic, summary);
        let traditional = PatsagiCouncilSimulator::review(&request);

        let (shapley, banzhaf) = Self::evaluate_patsagi_coalition(participants, coalition_value_fn);

        let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let shapley_variance = Self::shapley_variance(&shapley);
        let mercy_alignment = request.mercy_impact_score;

        let risk_score = Self::calculate_governance_risk_score(
            max_banzhaf, shapley_variance, mercy_alignment
        );

        let final_decision = if risk_score > 0.75 {
            PatsagiDecision::RequiresCouncilArbitration { councils: vec![13] }
        } else if risk_score > 0.55 {
            PatsagiDecision::RequiresSelfEvolution { priority: 2 }
        } else {
            traditional
        };

        let recommended_action = match final_decision {
            PatsagiDecision::RequiresCouncilArbitration { .. } => "Escalate to Council #13".to_string(),
            PatsagiDecision::RequiresSelfEvolution { .. } => "Trigger self-evolution".to_string(),
            _ => "No escalation".to_string(),
        };

        let report = GovernanceRiskReport {
            risk_score,
            max_banzhaf,
            shapley_variance,
            mercy_alignment,
            recommended_action,
        };

        (final_decision, report)
    }

    fn shapley_variance(shapley: &[(String, f64)]) -> f64 {
        if shapley.is_empty() { return 0.0; }
        let mean = shapley.iter().map(|(_, v)| *v).sum::<f64>() / shapley.len() as f64;
        shapley.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / shapley.len() as f64
    }

    pub fn apply_patsagi_decision(mesh: &mut DistributedMercyMesh, decision: &PatsagiDecision) -> String {
        match decision {
            PatsagiDecision::Approved { confidence } => format!("Approved ({:.2})", confidence),
            PatsagiDecision::RequiresSelfEvolution { priority } => format!("Self-evolution (priority {})", priority),
            PatsagiDecision::RequiresCouncilArbitration { councils } => format!("Arbitration {:?}", councils),
            PatsagiDecision::Rejected { reason, .. } => format!("Rejected: {}", reason),
        }
    }

    pub fn check_and_suggest_self_evolution(mesh: &DistributedMercyMesh) -> Option<String> {
        let report = Self::run_full_lattice_diagnostics(mesh);
        if !report.unified_organism_healthy || report.average_mercy_alignment < 0.92 {
            Some("Self-evolution recommended".to_string())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatticeDiagnosticsReport {
    pub unified_organism_healthy: bool,
    pub pending_healing_requests: usize,
    pub total_audit_entries: usize,
    pub average_mercy_alignment: f64,
    pub hybrid_channels_active: bool,
    pub overall_status: String,
}
