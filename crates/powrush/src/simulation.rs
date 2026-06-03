//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.13 — Interest Management Implementation
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0

/*!
# Interest Management Implementation (v16.13)

Interest Management is critical for MMO-scale performance.
Instead of updating every entity for every observer, we only process
entities that are "interesting" to a given player or Regional Council.

## Current Implementation (Lightweight but Production-Ready)

- Component-based `EntityStorage` with spatial query support
- Simple grid-based spatial partitioning inside each shard
- `InterestSet` per observer (player or council)
- Efficient range queries using the component layout

This is designed to work seamlessly with:
- `ShardManager` and per-zone `WorldSimulation`
- `CouncilProposal` routing (Regional councils benefit greatly)
- `MercyEvaluationSystem`

Future evolution: Quadtree / more advanced spatial structures when needed.
*/

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcType, NpcState, NpcManager};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== SHARD & COUNCIL TYPES ====================

pub type ShardId = u32;
pub type RegionId = u32;
pub type EntityId = u64;

#[derive(Debug, Clone)]
pub struct ShardContext {
    pub shard_id: ShardId,
    pub name: String,
    pub active_players: usize,
}

impl ShardContext {
    pub fn new(shard_id: ShardId, name: &str) -> Self {
        Self { shard_id, name: name.to_string(), active_players: 0 }
    }
}

// ==================== COUNCIL PROPOSAL PROTOCOL ====================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouncilScope {
    Global,
    Regional { shard_id: ShardId, region: Option<RegionId> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CouncilProposalType {
    AdjustHarmony { entity_id: EntityId, delta: f64 },
    IssueCommand(SimulationCommand),
    VetoOrModifyCommand { command_id: u64, modification: Option<SimulationCommand> },
    RequestScopedSnapshot { scope: SnapshotScope },
    ProposeStructuralChange { description: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotScope {
    Shard(ShardId),
    Region(ShardId, RegionId),
    Entity(EntityId),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalTarget {
    Global,
    Shard(ShardId),
    Region(ShardId, RegionId),
    Entity(EntityId),
    Command(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalPriority { Low, Normal, High, Critical }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilProposal {
    pub id: u64,
    pub council_id: String,
    pub scope: CouncilScope,
    pub proposal_type: CouncilProposalType,
    pub target: ProposalTarget,
    pub mercy_evaluation: MercyEvaluation,
    pub priority: ProposalPriority,
    pub reasoning: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDecision {
    pub proposal_id: u64,
    pub accepted: bool,
    pub applied_effects: Vec<String>,
    pub mercy_impact: f64,
    pub notes: Vec<String>,
}

// ==================== MERCY EVALUATION SYSTEM ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MercyGate {
    RadicalLove, BoundlessMercy, Service, Abundance, Truth, Joy, CosmicHarmony,
}

impl MercyGate {
    pub fn all() -> [MercyGate; 7] {
        [MercyGate::RadicalLove, MercyGate::BoundlessMercy, MercyGate::Service,
         MercyGate::Abundance, MercyGate::Truth, MercyGate::Joy, MercyGate::CosmicHarmony]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MercyGate::RadicalLove => "Radical Love",
            MercyGate::BoundlessMercy => "Boundless Mercy",
            MercyGate::Service => "Service",
            MercyGate::Abundance => "Abundance",
            MercyGate::Truth => "Truth",
            MercyGate::Joy => "Joy",
            MercyGate::CosmicHarmony => "Cosmic Harmony",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyEvaluation {
    pub overall_score: f64,
    pub gate_scores: HashMap<MercyGate, f64>,
    pub harmony_impact: f64,
    pub notes: Vec<String>,
}

impl MercyEvaluation {
    pub fn new() -> Self {
        Self { overall_score: 0.5, gate_scores: HashMap::new(), harmony_impact: 0.0, notes: vec![] }
    }
}

pub struct MercyEvaluationSystem;

impl MercyEvaluationSystem {
    pub fn evaluate_command(command: &SimulationCommand) -> MercyEvaluation { /* ... */ }
    fn score_against_gate(command: &SimulationCommand, gate: MercyGate) -> f64 { /* ... */ }
    pub fn apply_to_entity(storage: &mut EntityStorage, id: EntityId, eval: &MercyEvaluation) { /* ... */ }
}

// ==================== ENTITY STORAGE + INTEREST MANAGEMENT ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityType { pub entity_type: String }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionComponent { pub position: Position }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonyComponent { pub harmony: f64 }

/// Component-based entity storage with Interest Management support.
pub struct EntityStorage {
    pub entity_types: HashMap<EntityId, EntityType>,
    pub positions: HashMap<EntityId, PositionComponent>,
    pub harmonies: HashMap<EntityId, HarmonyComponent>,
    next_id: EntityId,
    // Simple spatial grid for interest management (cell size = 10.0 units)
    spatial_grid: HashMap<(i32, i32), Vec<EntityId>>,
}

impl EntityStorage {
    pub fn new() -> Self {
        Self {
            entity_types: HashMap::new(),
            positions: HashMap::new(),
            harmonies: HashMap::new(),
            next_id: 1000,
            spatial_grid: HashMap::new(),
        }
    }

    pub fn next_id(&mut self) -> EntityId { let id = self.next_id; self.next_id += 1; id }

    pub fn spawn(&mut self, entity_type: &str, position: Position, harmony: f64) -> EntityId {
        let id = self.next_id();
        self.entity_types.insert(id, EntityType { entity_type: entity_type.to_string() });
        self.positions.insert(id, PositionComponent { position });
        self.harmonies.insert(id, HarmonyComponent { harmony });

        // Update spatial grid
        let cell = self.position_to_cell(&position);
        self.spatial_grid.entry(cell).or_default().push(id);

        id
    }

    fn position_to_cell(&self, pos: &Position) -> (i32, i32) {
        ((pos.x / 10.0) as i32, (pos.y / 10.0) as i32)
    }

    pub fn get_position(&self, id: EntityId) -> Option<Position> { self.positions.get(&id).map(|p| p.position) }
    pub fn get_harmony(&self, id: EntityId) -> Option<f64> { self.harmonies.get(&id).map(|h| h.harmony) }

    pub fn set_position(&mut self, id: EntityId, new_pos: Position) {
        if let Some(old_pos) = self.positions.get(&id) {
            let old_cell = self.position_to_cell(old_pos);
            if let Some(vec) = self.spatial_grid.get_mut(&old_cell) {
                vec.retain(|&eid| eid != id);
            }
        }
        self.positions.insert(id, PositionComponent { position: new_pos });
        let new_cell = self.position_to_cell(&new_pos);
        self.spatial_grid.entry(new_cell).or_default().push(id);
    }

    pub fn set_harmony(&mut self, id: EntityId, harmony: f64) {
        if let Some(comp) = self.harmonies.get_mut(&id) {
            comp.harmony = harmony;
        }
    }

    pub fn remove(&mut self, id: EntityId) {
        if let Some(pos) = self.positions.remove(&id) {
            let cell = self.position_to_cell(&pos);
            if let Some(vec) = self.spatial_grid.get_mut(&cell) {
                vec.retain(|&eid| eid != id);
            }
        }
        self.entity_types.remove(&id);
        self.harmonies.remove(&id);
    }

    pub fn len(&self) -> usize { self.entity_types.len() }

    /// Interest Management query: returns entities within a radius of a position.
    /// Uses the spatial grid for efficiency.
    pub fn query_entities_in_range(&self, center: Position, radius: f32) -> Vec<EntityId> {
        let mut result = Vec::new();
        let cell_radius = (radius / 10.0).ceil() as i32;
        let center_cell = self.position_to_cell(&center);

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                if let Some(ids) = self.spatial_grid.get(&(center_cell.0 + dx, center_cell.1 + dy)) {
                    for &id in ids {
                        if let Some(pos) = self.positions.get(&id) {
                            let dx = pos.position.x - center.x;
                            let dy = pos.position.y - center.y;
                            if (dx*dx + dy*dy) <= (radius * radius) as f64 {
                                result.push(id);
                            }
                        }
                    }
                }
            }
        }
        result
    }
}

// ==================== INTEREST SET ====================

/// Represents what a player or Regional Council is currently interested in.
#[derive(Debug, Clone, Default)]
pub struct InterestSet {
    pub observer_position: Position,
    pub radius: f32,
    pub interested_entities: Vec<EntityId>,
}

impl InterestSet {
    pub fn new(position: Position, radius: f32) -> Self {
        Self { observer_position: position, radius, interested_entities: vec![] }
    }

    /// Update the interest set based on current world state.
    pub fn update(&mut self, storage: &EntityStorage) {
        self.interested_entities = storage.query_entities_in_range(self.observer_position, self.radius);
    }
}

// ==================== SIMULATION COMMAND ====================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulationCommand { /* ... */ }

// ==================== SHARD MANAGER ====================

pub struct ShardManager { /* ... with InterestSet support possible ... */ }

impl ShardManager {
    pub fn new() -> Self { /* ... */ }
    pub fn create_shard(&mut self, name: &str) -> ShardId { /* ... */ }
    pub fn route_proposal(&mut self, proposal: &CouncilProposal) -> Vec<CouncilDecision> { /* ... */ }
    // Future: integrate InterestSet per Regional Council
}

// ==================== WORLD SIMULATION ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerState { /* ... */ }

impl Default for PlayerState { /* ... */ }

pub struct WorldSimulation {
    pub shard_context: ShardContext,
    pub entities: EntityStorage,
    pub player: PlayerState,
    pub tick_count: u64,
    pub interest_set: InterestSet, // Per-shard interest tracking (can be per-player later)
}

impl WorldSimulation {
    pub fn new(shard_id: ShardId, name: &str) -> Self {
        Self {
            shard_context: ShardContext::new(shard_id, name),
            entities: EntityStorage::new(),
            player: PlayerState::default(),
            tick_count: 0,
            interest_set: InterestSet::new(Position { x: 0.0, y: 0.0 }, 50.0),
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.authoritative_tick(dt);
    }

    pub fn authoritative_tick(&mut self, _dt: f32) {
        self.tick_count += 1;
        // Update interest set (example: around player)
        self.interest_set.observer_position = self.player.position;
        self.interest_set.update(&self.entities);
    }

    pub fn evaluate_command_with_mercy(&mut self, command: &SimulationCommand, target_entity: Option<EntityId>) {
        let evaluation = MercyEvaluationSystem::evaluate_command(command);
        if let Some(id) = target_entity {
            MercyEvaluationSystem::apply_to_entity(&mut self.entities, id, &evaluation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interest_management_basic() {
        let mut storage = EntityStorage::new();
        let id1 = storage.spawn("Player", Position { x: 5.0, y: 5.0 }, 0.8);
        let id2 = storage.spawn("NPC", Position { x: 12.0, y: 5.0 }, 0.6);

        let nearby = storage.query_entities_in_range(Position { x: 5.0, y: 5.0 }, 10.0);
        assert!(nearby.contains(&id1));
        assert!(nearby.contains(&id2));
    }
}