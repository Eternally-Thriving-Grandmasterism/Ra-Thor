//! Distributed Mercy Mesh — v14.0.7 Thunder Lattice
//! Event-driven mercy propagation + governance hooks + networking stubs.

use std::collections::HashMap;

/// Core event types including governance and future networking events.
#[derive(Debug, Clone)]
pub enum MercyEvent {
    HealingTriggered { severity: f64, organism_id: String },
    ConvictionStakeUpdated { proposal_id: String, staker_id: String, new_conviction: f64 },
    GovernanceVoteCast { proposal_id: String, voter_id: String, effective_power: f64 },
    SelfEvolutionProposalSubmitted { proposal_id: String, mercy_alignment: f64 },
    GovernanceCycleCompleted { proposal_id: String, passed: bool, final_score: f64 },
    // Networking stubs for future sovereign channels
    SovereignChannelOpened { from: String, to: String },
    MeshMessageReceived { from: String, payload_type: String },
}

#[derive(Debug, Clone)]
pub struct MercyMeshConfig {
    pub mercy_threshold_for_governance_trigger: f64,
    pub enable_governance_hooks: bool,
    pub enable_networking_stubs: bool,
}

impl Default for MercyMeshConfig {
    fn default() -> Self {
        Self {
            mercy_threshold_for_governance_trigger: 0.75,
            enable_governance_hooks: true,
            enable_networking_stubs: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrganismNode {
    pub id: String,
    pub mercy_capacity: f64,
}

/// Production-grade Distributed Mercy Mesh with governance + networking foundation.
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
        Self { nodes: HashMap::new(), config, event_log: Vec::new() }
    }

    pub fn register_organism(&mut self, node: OrganismNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn propagate_mercy_event(&mut self, event: MercyEvent) {
        self.event_log.push(event.clone());

        if self.config.enable_governance_hooks {
            if let MercyEvent::HealingTriggered { severity, .. } = &event {
                if *severity >= self.config.mercy_threshold_for_governance_trigger {
                    println!("[MERCY MESH] High mercy event — governance opportunity triggered");
                }
            }
        }
    }

    pub fn emit_governance_event(&mut self, event: MercyEvent) {
        self.event_log.push(event.clone());
        println!("[MERCY MESH] Governance event emitted");
    }

    // === Networking Stubs (for future sovereign channel implementation) ===
    pub fn open_sovereign_channel(&mut self, from: &str, to: &str) {
        if self.config.enable_networking_stubs {
            self.event_log.push(MercyEvent::SovereignChannelOpened {
                from: from.to_string(),
                to: to.to_string(),
            });
            println!("[MERCY MESH] Sovereign channel stub opened: {} -> {}", from, to);
        }
    }

    pub fn receive_mesh_message(&mut self, from: &str, payload_type: &str) {
        if self.config.enable_networking_stubs {
            self.event_log.push(MercyEvent::MeshMessageReceived {
                from: from.to_string(),
                payload_type: payload_type.to_string(),
            });
            println!("[MERCY MESH] Mesh message received (stub): {} - {}", from, payload_type);
        }
    }

    pub fn on_healing_may_trigger_governance(&self, severity: f64) -> bool {
        severity >= self.config.mercy_threshold_for_governance_trigger
    }

    pub fn get_recent_events(&self, limit: usize) -> Vec<MercyEvent> {
        self.event_log.iter().rev().take(limit).cloned().collect()
    }
}