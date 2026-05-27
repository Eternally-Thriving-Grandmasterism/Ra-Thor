// crates/lattice-conductor-v14/src/lattice_conductor_enhancements.rs
// Integration of CooperativeGame into PATSAGi flow

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

    // ==================== PATSAGi + Cooperative Game Theory Integration ====================

    /// Creates a PATSAGi review request (existing)
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

    /// NEW: Evaluate a PATSAGi-style coalition using Cooperative Game Theory
    /// Returns (Shapley Values, Banzhaf Index)
    pub fn evaluate_patsagi_coalition(
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
        let game = CooperativeGame::new(participants, coalition_value_fn);
        let shapley = game.shapley_value();
        let banzhaf = game.banzhaf_index();
        (shapley, banzhaf)
    }

    /// NEW: Full PATSAGi flow with game-theoretic analysis
    pub fn submit_to_patsagi_with_game_theory(
        mesh: &mut DistributedMercyMesh,
        topic: &str,
        summary: &str,
        participants: Vec<String>,
        coalition_value_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static,
    ) -> String {
        let request = Self::request_patsagi_review(mesh, topic, summary);

        // Get traditional PATSAGi decision
        let traditional_decision = PatsagiCouncilSimulator::review(&request);

        // Get game-theoretic analysis
        let (shapley, banzhaf) = Self::evaluate_patsagi_coalition(participants, coalition_value_fn);

        // Combine insights (simple version for now)
        let game_insight = format!(
            "Shapley: {:?} | Banzhaf: {:?}",
            shapley.iter().map(|(p, v)| format!("{}:{:.2}", p, v)).collect::<Vec<_>>(),
            banzhaf.iter().map(|(p, v)| format!("{}:{:.3}", p, v)).collect::<Vec<_>>()
        );

        format!(
            "Traditional: {} | Game Theory: {}",
            LatticeConductorEnhancements::apply_patsagi_decision(mesh, &traditional_decision),
            game_insight
        )
    }

    pub fn apply_patsagi_decision(mesh: &mut DistributedMercyMesh, decision: &PatsagiDecision) -> String {
        // existing implementation...
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
