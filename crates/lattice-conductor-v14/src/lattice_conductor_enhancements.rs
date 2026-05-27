// crates/lattice-conductor-v14/src/lattice_conductor_enhancements.rs
// v14.1 Lattice Conductor Enhancements
// PATSAGi Runtime Hooks Implementation

use crate::distributed_mercy_mesh::{DistributedMercyMesh, MercyEvent, OrganismNode};

/// PATSAGi Council review request
#[derive(Debug, Clone)]
pub struct PatsagiReviewRequest {
    pub topic: String,
    pub summary: String,
    pub mercy_impact_score: f64,
    pub requested_by: String,
}

/// Decision returned by PATSAGi Council simulation / integration
#[derive(Debug, Clone, PartialEq)]
pub enum PatsagiDecision {
    Approved,
    RequiresSelfEvolution,
    RequiresCouncilArbitration,
    Rejected { reason: String },
}

pub struct LatticeConductorEnhancements;

impl LatticeConductorEnhancements {
    // ... existing methods ...

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
        let pending_requests = mesh.get_pending_requests().len();
        let audit_entries = mesh.get_audit_log().len();

        let avg_mercy_score = if audit_entries > 0 {
            let sum: f64 = mesh.get_audit_log().iter().map(|a| a.mercy_score).sum();
            sum / audit_entries as f64
        } else { 0.999 };

        LatticeDiagnosticsReport {
            unified_organism_healthy: unified_healthy,
            pending_healing_requests: pending_requests,
            total_audit_entries: audit_entries,
            average_mercy_alignment: avg_mercy_score,
            hybrid_channels_active: true,
            overall_status: if unified_healthy && avg_mercy_score > 0.9 {
                "Healthy - ONE Organism thriving".to_string()
            } else {
                "Degraded - Review required".to_string()
            },
        }
    }

    pub fn propagate_audit_to_mesh(mesh: &mut DistributedMercyMesh, _action: &str, mercy_score: f64) {
        let event = MercyEvent::ConvictionUpdated {
            organism_id: "lattice-conductor".to_string(),
            new_score: mercy_score,
        };
        mesh.propagate_mercy_event(event);
    }

    pub fn trigger_geometric_healing_cycle(mesh: &mut DistributedMercyMesh, severity: f64) {
        Self::enforce_one_organism_identity(mesh);
        mesh.propagate_mercy_event(MercyEvent::HealingTriggered {
            severity,
            organism_id: Some("lattice-conductor".to_string()),
        });
    }

    /// NEW: Request review from PATSAGi Councils
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

    /// NEW: Apply PATSAGi Council decision (runtime hook)
    pub fn apply_patsagi_decision(
        mesh: &mut DistributedMercyMesh,
        decision: &PatsagiDecision,
    ) -> String {
        match decision {
            PatsagiDecision::Approved => {
                "PATSAGi approved. Continuing normal operations.".to_string()
            }
            PatsagiDecision::RequiresSelfEvolution => {
                if let Some(suggestion) = Self::check_and_suggest_self_evolution(mesh) {
                    format!("PATSAGi decision: Self-evolution triggered. {}", suggestion)
                } else {
                    "PATSAGi requested self-evolution.".to_string()
                }
            }
            PatsagiDecision::RequiresCouncilArbitration => {
                "Escalating to full PATSAGi Council arbitration.".to_string()
            }
            PatsagiDecision::Rejected { reason } => {
                format!("PATSAGi rejected action. Reason: {}", reason)
            }
        }
    }

    pub fn check_and_suggest_self_evolution(mesh: &DistributedMercyMesh) -> Option<String> {
        let report = Self::run_full_lattice_diagnostics(mesh);

        if !report.unified_organism_healthy || report.average_mercy_alignment < 0.92 {
            Some(format!(
                "Self-evolution recommended. Status: {}. Mercy alignment: {:.3}",
                report.overall_status, report.average_mercy_alignment
            ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_mercy_mesh::DistributedMercyMesh;

    #[test]
    fn test_patsagi_hooks() {
        let mut mesh = DistributedMercyMesh::new();
        let request = LatticeConductorEnhancements::request_patsagi_review(
            &mesh,
            "Major lattice diagnostic",
            "Routine health check",
        );
        assert_eq!(request.requested_by, "lattice-conductor");

        let decision = PatsagiDecision::Approved;
        let result = LatticeConductorEnhancements::apply_patsagi_decision(&mut mesh, &decision);
        assert!(result.contains("approved"));
    }
}