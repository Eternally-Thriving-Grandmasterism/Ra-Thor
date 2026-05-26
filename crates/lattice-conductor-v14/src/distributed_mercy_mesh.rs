//! Distributed Mercy Mesh — v14.0.6 Thunder Lattice
//! Event-driven mercy propagation across organisms with governance hooks.

use std::collections::HashMap;

/// Core event types in the Distributed Mercy Mesh, including governance triggers.
#[derive(Debug, Clone)]
pub enum MercyEvent {
    HealingTriggered { severity: f64, organism_id: String },
    ConvictionStakeUpdated { proposal_id: String, staker_id: String, new_conviction: f64 },
    GovernanceVoteCast { proposal_id: String, voter_id: String, effective_power: f64 },
    SelfEvolutionProposalSubmitted { proposal_id: String, mercy_alignment: f64 },
    GovernanceCycleCompleted { proposal_id: String, passed: bool, final_score: f64 },
}

/// Configuration for the mercy mesh.
#[derive(Debug, Clone)]
pub struct MercyMeshConfig {
    pub mercy_threshold_for_governance_trigger: f64,
    pub enable_governance_hooks: bool,
}

impl Default for MercyMeshConfig {
    fn default() -> Self {
        Self {
            mercy_threshold_for_governance_trigger: 0.75,
            enable_governance_hooks: true,
        }
    }
}

/// Represents a participating organism in the mesh.
#[derive(Debug, Clone)]
pub struct OrganismNode {
    pub id: String,
    pub mercy_capacity: f64,
}

/// Distributed Mercy Mesh with governance event hooks.
pub struct DistributedMercyMesh {
    nodes: HashMap<String, OrganismNode>,
    config: MercyMeshConfig,
    event_log: Vec<MercyEvent>,
}

impl DistributedMercyMesh {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            config: MercyMeshConfig::default(),
            event_log: Vec::new(),
        }
    }

    pub fn with_config(config: MercyMeshConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            config,
            event_log: Vec::new(),
        }
    }

    pub fn register_organism(&mut self, node: OrganismNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Core propagation — now with optional governance hook triggering.
    pub fn propagate_mercy_event(&mut self, event: MercyEvent) {
        self.event_log.push(event.clone());

        if self.config.enable_governance_hooks {
            if let MercyEvent::HealingTriggered { severity, .. } = &event {
                if *severity >= self.config.mercy_threshold_for_governance_trigger {
                    // Trigger governance opportunity from healing event
                    println!("[MERCY MESH] High-severity healing event triggered governance hook");
                    // In full system this would emit a governance proposal opportunity
                }
            }
        }
    }

    /// New: Emit a governance-related event into the mesh.
    pub fn emit_governance_event(&mut self, event: MercyEvent) {
        match &event {
            MercyEvent::ConvictionStakeUpdated { .. } |
            MercyEvent::GovernanceVoteCast { .. } |
            MercyEvent::SelfEvolutionProposalSubmitted { .. } |
            MercyEvent::GovernanceCycleCompleted { .. } => {
                self.event_log.push(event.clone());
                println!("[MERCY MESH] Governance event propagated: {:?}", event);
            }
            _ => {}
        }
    }

    /// When a healing event is strong enough, offer to create a governance opportunity.
    pub fn on_healing_may_trigger_governance(&self, severity: f64) -> bool {
        severity >= self.config.mercy_threshold_for_governance_trigger
    }

    pub fn get_recent_events(&self, limit: usize) -> Vec<MercyEvent> {
        self.event_log.iter().rev().take(limit).cloned().collect()
    }
}