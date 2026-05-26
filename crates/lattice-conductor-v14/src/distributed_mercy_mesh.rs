//! Distributed Mercy Mesh v14.0.8+ — Hybrid Classical + Post-Quantum Channel Support

use crate::hybrid_sovereign_channel::{HybridSovereignChannel, HybridSovereignChannelManager};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum MercyEvent {
    HealingTriggered { severity: f64, organism_id: String },
    HybridChannelOpened { from: String, to: String },
    EncryptedMessageReceived { from: String, channel_id: String },
}

#[derive(Debug, Clone)]
pub struct MercyMeshConfig {
    pub enable_hybrid_channels: bool,
}

impl Default for MercyMeshConfig {
    fn default() -> Self {
        Self { enable_hybrid_channels: true }
    }
}

pub struct OrganismNode {
    pub id: String,
    pub mercy_capacity: f64,
}

/// Distributed Mercy Mesh with Hybrid (Classical + Post-Quantum) channel support.
pub struct DistributedMercyMesh {
    nodes: HashMap<String, OrganismNode>,
    config: MercyMeshConfig,
    event_log: Vec<MercyEvent>,
    hybrid_channels: HybridSovereignChannelManager,
}

impl DistributedMercyMesh {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            config: MercyMeshConfig::default(),
            event_log: Vec::new(),
            hybrid_channels: HybridSovereignChannelManager::new(),
        }
    }

    pub fn register_organism(&mut self, node: OrganismNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Create a hybrid (AES-GCM + Post-Quantum KEM) sovereign channel.
    pub fn create_hybrid_channel(
        &mut self,
        from: &str,
        to: &str,
    ) -> Option<&mut HybridSovereignChannel> {
        if !self.config.enable_hybrid_channels {
            return None;
        }
        let channel = self.hybrid_channels.create_channel(from, to);
        self.event_log.push(MercyEvent::HybridChannelOpened {
            from: from.to_string(),
            to: to.to_string(),
        });
        Some(channel)
    }

    pub fn propagate_mercy_event(&mut self, event: MercyEvent) {
        self.event_log.push(event.clone());
    }
}