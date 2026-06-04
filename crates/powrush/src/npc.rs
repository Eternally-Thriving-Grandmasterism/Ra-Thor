//! crates/powrush/src/npc.rs
//! Powrush NPC System — Hybrid AI + Mercy-Gated Behavior
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0

/*!
# NPC Module (Stub / Foundation)

This module provides the foundation for Powrush's hybrid NPC AI system.
It is designed to integrate with:

- `ShardManager` and per-zone `WorldSimulation`
- `CouncilProposal` routing (Regional councils can influence NPCs)
- `MercyEvaluationSystem` (NPC actions can be evaluated)
- `EntityStorage` (NPCs as first-class entities with components)

## Current Status (v16.11+)

This is currently a clean foundation/stub. Full hybrid NPC AI logic
(from earlier `powrush_mmo_core` and v15 work) can be re-integrated here
or kept in `powrush_mmo_core.rs` depending on architectural preference.

PATSAGi Council guidance: NPCs should be mercy-evaluable and
influenceable by Regional Councils.
*/

use crate::simulation::{EntityId, Position, ShardId};
use std::collections::HashMap;

/// Basic NPC type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NpcType {
    Basic,
    Merchant,
    Guardian,
    QuestGiver,
    Custom(String),
}

/// Lightweight NPC state for use inside `EntityStorage` and shards
#[derive(Debug, Clone)]
pub struct NpcState {
    pub npc_type: NpcType,
    pub position: Position,
    pub harmony: f64,
    pub shard_id: Option<ShardId>,
}

impl NpcState {
    pub fn new(npc_type: NpcType, position: Position) -> Self {
        Self {
            npc_type,
            position,
            harmony: 0.6,
            shard_id: None,
        }
    }
}

/// Simple NPC manager (can be expanded or replaced by full system in powrush_mmo_core)
pub struct NpcManager {
    pub npcs: HashMap<EntityId, NpcState>,
}

impl NpcManager {
    pub fn new() -> Self {
        Self { npcs: HashMap::new() }
    }

    pub fn spawn_npc(&mut self, id: EntityId, npc_type: NpcType, position: Position) -> EntityId {
        let state = NpcState::new(npc_type, position);
        self.npcs.insert(id, state);
        id
    }

    pub fn get(&self, id: EntityId) -> Option<&NpcState> {
        self.npcs.get(&id)
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut NpcState> {
        self.npcs.get_mut(&id)
    }

    pub fn remove(&mut self, id: EntityId) {
        self.npcs.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npc_manager_basic() {
        let mut manager = NpcManager::new();
        let id = manager.spawn_npc(100, NpcType::Basic, Position { x: 0.0, y: 0.0 });
        assert!(manager.get(id).is_some());
    }
}