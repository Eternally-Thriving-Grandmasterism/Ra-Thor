//! Distributed Mercy Mesh v14.0.7+ — Symbiotic integration with Sovereign Encrypted Channels

use crate::sovereign_channel::{SovereignChannel, SovereignChannelManager, ChannelDirection};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum MercyEvent {
    HealingTriggered { severity: f64, organism_id: String },
    ConvictionStakeUpdated { proposal_id: String, staker_id: String, new_conviction: f64 },
    GovernanceVoteCast { proposal_id: String, voter_id: String, effective_power: f64 },
    SelfEvolutionProposalSubmitted { proposal_id: String, mercy_alignment: f64 },
    GovernanceCycleCompleted { proposal_id: String, passed: bool, final_score: f64 },
    SovereignChannelOpened { from: String, to: String },
    EncryptedMessageReceived { from: String, channel_id: String },
}

#[derive(Debug, Clone)]
pub struct MercyMeshConfig {
    pub mercy_threshold_for_governance_trigger: f64,
    pub enable_governance_hooks: bool,
    pub enable_encrypted_channels: bool,
}

impl Default for MercyMeshConfig {
    fn default() -> Self {
        Self {
            mercy_threshold_for_governance_trigger: 0.75,
            enable_governance_hooks: true,
            enable_encrypted_channels: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrganismNode {
    pub id: String,
    pub mercy_capacity: f64,
}

/// Distributed Mercy Mesh with symbiotic Sovereign Encrypted Channel support.
pub struct DistributedMercyMesh {
    nodes: HashMap<String, OrganismNode>,
    config: MercyMeshConfig,
    event_log: Vec<MercyEvent>,
    encrypted_channels: SovereignChannelManager,
}

impl DistributedMercyMesh {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            config: MercyMeshConfig::default(),
            event_log: Vec::new(),
            encrypted_channels: SovereignChannelManager::new(),
        }
    }

    pub fn register_organism(&mut self, node: OrganismNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Create and register a new encrypted sovereign channel between two organisms.
    pub fn create_encrypted_sovereign_channel(
        &mut self,
        from: &str,
        to: &str,
    ) -> Option<&mut SovereignChannel> {
        if !self.config.enable_encrypted_channels {
            return None;
        }

        let channel = self.encrypted_channels.create_channel(from, to, ChannelDirection::Bidirectional);

        self.event_log.push(MercyEvent::SovereignChannelOpened {
            from: from.to_string(),
            to: to.to_string(),
        });

        println!("[MERCY MESH] Created encrypted sovereign channel between {} and {}", from, to);
        Some(channel)
    }

    pub fn propagate_mercy_event(&mut self, event: MercyEvent) {
        self.event_log.push(event.clone());
    }

    /// Route an encrypted message through the mesh using sovereign channels.
    pub fn route_encrypted_message(
        &mut self,
        channel_id: &str,
        payload: &[u8],
    ) -> Option<MercyEvent> {
        println!("[MERCY MESH] Routing encrypted message via channel: {}", channel_id);
        Some(MercyEvent::EncryptedMessageReceived {
            from: "mesh".to_string(),
            channel_id: channel_id.to_string(),
        })
    }

    pub fn get_recent_events(&self, limit: usize) -> Vec<MercyEvent> {
        self.event_log.iter().rev().take(limit).cloned().collect()
    }
}