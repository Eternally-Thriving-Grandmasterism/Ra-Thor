// crates/lattice-conductor-v14/src/lattice_conductor_enhancements.rs
// Symbiotic extension: PATSAGi + Thunder Lattice with Cooperative Game Theory

use crate::distributed_mercy_mesh::{DistributedMercyMesh, MercyEvent, OrganismNode};
use crate::patsagi_governance::{PatsagiCouncilSimulator, PatsagiDecision, PatsagiReviewRequest};
use crate::cooperative_governance::CooperativeGame;
use std::collections::HashSet;

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

    /// Refined PATSAGi + Game Theory (influences decision)
    pub fn submit_to_patsagi_with_game_theory(
        mesh: &mut DistributedMercyMesh,
        topic: &str,
        summary: &str,
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> (PatsagiDecision, String) {
        let request = Self::request_patsagi_review(mesh, topic, summary);
        let traditional = PatsagiCouncilSimulator::review(&request);

        let (shapley, banzhaf) = Self::evaluate_patsagi_coalition(participants, coalition_value_fn);

        let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let power_concentrated = max_banzhaf > 0.6;

        let shapley_variance = Self::shapley_variance(&shapley);
        let unfair_contribution = shapley_variance > 0.15;

        let final_decision = match (&traditional, power_concentrated, unfair_contribution) {
            (PatsagiDecision::Approved { .. }, true, _) => {
                PatsagiDecision::RequiresCouncilArbitration { councils: vec![13] }
            }
            (PatsagiDecision::Approved { .. }, _, true) => {
                PatsagiDecision::RequiresSelfEvolution { priority: 2 }
            }
            _ => traditional,
        };

        let insight = format!(
            "Shapley variance: {:.3} | Max Banzhaf: {:.3} | Power concentrated: {} | Unfair: {}",
            shapley_variance, max_banzhaf, power_concentrated, unfair_contribution
        );

        (final_decision, insight)
    }

    fn shapley_variance(shapley: &[(String, f64)]) -> f64 {
        if shapley.is_empty() { return 0.0; }
        let mean = shapley.iter().map(|(_, v)| *v).sum::<f64>() / shapley.len() as f64;
        shapley.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / shapley.len() as f64
    }

    // ==================== Symbiotic Thunder Lattice Extension ====================

    /// NEW: Thunder Lattice vote with cooperative game theory influence
    pub fn evaluate_thunder_lattice_vote(
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> (f64, String) {
        let game = CooperativeGame::new(participants.clone(), coalition_value_fn);
        let shapley = game.shapley_value();
        let banzhaf = game.banzhaf_index();

        // Use advanced voting as base, then adjust with game theory
        let base_score = Self::advanced_mercy_vote_tally(
            &participants.iter().enumerate().map(|(i, _)| {
                let mercy = 0.9; // placeholder
                let conviction = 100u64;
                (mercy, 1.0, conviction)
            }).collect::<Vec<_>>(),
            true,
        );

        let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let power_risk = if max_banzhaf > 0.55 { 0.15 } else { 0.0 };

        let adjusted_score = (base_score - power_risk).clamp(0.0, 1.0);

        let insight = format!(
            "Base: {:.3} | Adjusted: {:.3} | Max Banzhaf: {:.3}",
            base_score, adjusted_score, max_banzhaf
        );

        (adjusted_score, insight)
    }

    pub fn advanced_mercy_vote_tally(
        votes: &[(f64, f64, u64)],
        use_quadratic: bool,
    ) -> f64 {
        let mut total = 0.0;
        let mut weight_sum = 0.0;

        for (mercy, base, conviction) in votes {
            let conviction_factor = 1.0 + (*conviction as f64).ln_1p() * 0.15;
            let mut w = base * conviction_factor;
            if use_quadratic {
                w = w.sqrt();
            }
            total += mercy * w;
            weight_sum += w;
        }

        if weight_sum > 0.0 { total / weight_sum } else { 0.0 }
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
