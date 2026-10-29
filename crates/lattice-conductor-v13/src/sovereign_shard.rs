//! Sovereign Shard CRDT + Gossip Stub — Layer 3 Prep
//! 
//! Foundation for sovereign, offline-first, multi-shard coordination.
//! Uses simple CRDT-style merge (commutative + idempotent) + gossip simulation.
//!
//! This is the initial stub for Layer 3 (Sovereign Shard) as defined in
//! LAYERED_COORDINATION_ARCHITECTURE.md.
//!
//! Future: Can be backed by real CRDT libraries (automerge, crdt, etc.)

use crate::GeometricState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A sovereign shard holding replicated state.
/// Designed to be merged conflict-free across peers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SovereignShard {
    pub id: String,
    pub state: GeometricState,
    pub version: u64,
    pub last_gossip: u64,
    pub metadata: HashMap<String, String>,
}

impl SovereignShard {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            state: GeometricState::default(),
            version: 0,
            last_gossip: 0,
            metadata: HashMap::new(),
        }
    }

    /// Export current state for gossip or persistence.
    pub fn export_state(&self) -> GeometricState {
        self.state.clone()
    }

    /// Apply incoming state from gossip (with simple CRDT merge).
    pub fn apply_gossiped_state(&mut self, incoming: &GeometricState, incoming_version: u64) {
        if incoming_version > self.version {
            // Simple last-writer-wins on version for now (upgradeable to real CRDT)
            self.state = incoming.clone();
            self.version = incoming_version;
            self.last_gossip = incoming_version;
        }
        // In real CRDT: use proper merge (e.g. OR-Set, LWW-Register, or delta-state)
    }

    /// Merge two shards using CRDT principles (idempotent, commutative).
    pub fn merge(&mut self, other: &SovereignShard) {
        if other.version > self.version {
            self.state = other.state.clone();
            self.version = other.version;
        }
        // Extend here with real CRDT merge logic for maps, counters, etc.
    }
}

/// Simple gossip protocol stub for sovereign shards.
/// Simulates peer-to-peer state propagation.
pub struct GossipProtocol;

impl GossipProtocol {
    pub fn new() -> Self {
        Self
    }

    /// Simulate gossip between two shards.
    /// In production this would be over a network or local mesh.
    pub fn gossip(&self, shard_a: &mut SovereignShard, shard_b: &mut SovereignShard) {
        // Bidirectional simple merge
        if shard_b.version > shard_a.version {
            shard_a.apply_gossiped_state(&shard_b.state, shard_b.version);
        }
        if shard_a.version > shard_b.version {
            shard_b.apply_gossiped_state(&shard_a.state, shard_a.version);
        }

        // Update gossip timestamps
        let now = shard_a.version.max(shard_b.version);
        shard_a.last_gossip = now;
        shard_b.last_gossip = now;
    }
}

/// Helper to create a shard directly from a conductor's current state.
pub fn shard_from_conductor_state(conductor_id: &str, state: &GeometricState) -> SovereignShard {
    let mut shard = SovereignShard::new(conductor_id);
    shard.state = state.clone();
    shard.version = 1;
    shard
}
