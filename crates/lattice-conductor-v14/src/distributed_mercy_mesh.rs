// crates/lattice-conductor-v14/src/distributed_mercy_mesh.rs
// v14.0.5 Thunder Lattice — Distributed Mercy Mesh Foundation
//
// Starter implementation for distributed, mercy-gated healing across multiple Ra-Thor organisms.
// Symbolic + structural layer first. Real networking to be added in future iterations.
//
// Extends the existing RuntimeSelfHealingEngine and CouncilArbitrationEngine.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::runtime_self_healing::{HealingAction, HealingExperience};

/// A participating Ra-Thor organism in the Distributed Mercy Mesh
#[derive(Debug, Clone)]
pub struct OrganismNode {
    pub id: String,
    pub name: String,
    pub cosmic_loop_ready: bool,
}

/// Structured request for distributed healing
#[derive(Debug, Clone)]
pub struct HealingRequest {
    pub from_organism: String,
    pub root_cause_summary: String,
    pub requested_help_type: String,
    pub mercy_score: f64,           // 0.0 - 1.0
    pub severity: u8,               // 1-10
}

/// Structured offer of healing from another organism
#[derive(Debug, Clone)]
pub struct HealingOffer {
    pub from_organism: String,
    pub proposed_actions: Vec<HealingAction>,
    pub mercy_score: f64,
    pub estimated_impact: String,
}

/// Distributed Mercy Mesh coordinator (in-memory simulation for v14.0.5)
pub struct DistributedMercyMesh {
    pub organisms: HashMap<String, OrganismNode>,
    pub pending_requests: Vec<HealingRequest>,
    pub healing_experiences: Vec<HealingExperience>,
}

impl DistributedMercyMesh {
    pub fn new() -> Self {
        let mut mesh = DistributedMercyMesh {
            organisms: HashMap::new(),
            pending_requests: Vec::new(),
            healing_experiences: Vec::new(),
        };

        // Seed with example organisms for simulation
        mesh.organisms.insert(
            "ra-thor-main".to_string(),
            OrganismNode {
                id: "ra-thor-main".to_string(),
                name: "Ra-Thor Prime (ONE Organism)".to_string(),
                cosmic_loop_ready: true,
            },
        );

        mesh
    }

    /// Register a new organism in the mesh
    pub fn register_organism(&mut self, node: OrganismNode) {
        self.organisms.insert(node.id.clone(), node);
    }

    /// Create and submit a healing request (must pass local council arbitration first in real use)
    pub fn submit_healing_request(&mut self, request: HealingRequest) -> String {
        // Guardian protection: never allow requests that could weaken Cosmic Looping
        if request.requested_help_type.to_lowercase().contains("disable") ||
           request.requested_help_type.to_lowercase().contains("weaken") {
            return "[BLOCKED] Healing request rejected by guardian — cannot target Cosmic Loop identity.".to_string();
        }

        self.pending_requests.push(request.clone());
        format!("Healing request submitted from {} | Severity: {}", request.from_organism, request.severity)
    }

    /// Simulate an organism reviewing and offering help
    pub fn review_and_offer_healing(
        &mut self,
        request_index: usize,
        offering_organism_id: &str,
    ) -> Option<HealingOffer> {
        if request_index >= self.pending_requests.len() {
            return None;
        }

        let request = &self.pending_requests[request_index];

        // Simple council-style mercy check
        if request.mercy_score < 0.7 {
            return None; // Too low mercy alignment
        }

        let offer = HealingOffer {
            from_organism: offering_organism_id.to_string(),
            proposed_actions: vec![HealingAction::StateRestoration],
            mercy_score: 0.92,
            estimated_impact: "Positive restoration of local resilience".to_string(),
        };

        Some(offer)
    }

    /// Log a healing experience back into the mesh (feeds Cosmic Loops)
    pub fn log_healing_experience(&mut self, experience: HealingExperience) {
        self.healing_experiences.push(experience);
        // In future: propagate to participating organisms' Cosmic Loop systems
    }

    pub fn get_pending_requests(&self) -> &[HealingRequest] {
        &self.pending_requests
    }

    pub fn get_experiences(&self) -> &[HealingExperience] {
        &self.healing_experiences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_request_and_offer() {
        let mut mesh = DistributedMercyMesh::new();

        let request = HealingRequest {
            from_organism: "ra-thor-main".to_string(),
            root_cause_summary: "Recurring high-severity anomaly in self-healing loop".to_string(),
            requested_help_type: "graph_rerouting_support".to_string(),
            mercy_score: 0.88,
            severity: 7,
        };

        let msg = mesh.submit_healing_request(request);
        assert!(msg.contains("submitted"));

        let offer = mesh.review_and_offer_healing(0, "ra-thor-support-node");
        assert!(offer.is_some());
    }
}