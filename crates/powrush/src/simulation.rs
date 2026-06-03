//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.9 — Shard-Aware Foundations + Interest Management Exploration
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0 | Full backward compatible evolution

/*!
# Powrush Sharded Architecture Foundations (v16.9)

This version adds basic shard-aware structures to `WorldSimulation` and prepares
the ground for Interest Management on top of the component-based `EntityStorage`.

See `docs/powrush/sharding-architecture.md` for the full zone-based sharding design.
*/

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// Re-export shard types from architecture
pub use crate::simulation::ShardId; // Will be defined below for now

// ==================== SHARD-AWARE TYPES ====================

pub type ShardId = u32;
pub type RegionId = u32;

/// Basic context for a shard/zone.
#[derive(Debug, Clone)]
pub struct ShardContext {
    pub shard_id: ShardId,
    pub name: String,
    pub active_players: usize,
}

impl ShardContext {
    pub fn new(shard_id: ShardId, name: &str) -> Self {
        Self {
            shard_id,
            name: name.to_string(),
            active_players: 0,
        }
    }
}

// ==================== COUNCIL PROPOSAL PROTOCOL (from v16.8) ====================

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
pub enum ProposalPriority {
    Low,
    Normal,
    High,
    Critical,
}

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
    RadicalLove,
    BoundlessMercy,
    Service,
    Abundance,
    Truth,
    Joy,
    CosmicHarmony,
}

impl MercyGate {
    pub fn all() -> [MercyGate; 7] {
        [
            MercyGate::RadicalLove,
            MercyGate::BoundlessMercy,
            MercyGate::Service,
            MercyGate::Abundance,
            MercyGate::Truth,
            MercyGate::Joy,
            MercyGate::CosmicHarmony,
        ]
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
        Self {
            overall_score: 0.5,
            gate_scores: HashMap::new(),
            harmony_impact: 0.0,
            notes: Vec::new(),
        }
    }
}

pub struct MercyEvaluationSystem;

impl MercyEvaluationSystem {
    pub fn evaluate_command(command: &SimulationCommand) -> MercyEvaluation {
        let mut eval = MercyEvaluation::new();
        let mut total = 0.0;
        let mut count = 0;

        for gate in MercyGate::all() {
            let score = Self::score_against_gate(command, gate);
            eval.gate_scores.insert(gate, score);
            total += score;
            count += 1;
        }

        eval.overall_score = if count > 0 { total / count as f64 } else { 0.5 };
        eval.harmony_impact = (eval.overall_score - 0.5) * 0.1;

        if eval.overall_score > 0.75 {
            eval.notes.push("Strong mercy alignment".to_string());
        } else if eval.overall_score < 0.4 {
            eval.notes.push("Low mercy alignment".to_string());
        }

        eval
    }

    fn score_against_gate(command: &SimulationCommand, gate: MercyGate) -> f64 {
        match (command, gate) {
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::BoundlessMercy) => 0.9,
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::RadicalLove) => 0.85,
            (SimulationCommand::SpawnNpc { .. }, MercyGate::Service) => 0.7,
            (SimulationCommand::TradeWithNpc { .. }, MercyGate::Service) => 0.7,
            _ => 0.55,
        }
    }

    pub fn apply_to_entity(storage: &mut EntityStorage, id: EntityId, eval: &MercyEvaluation) {
        if let Some(current) = storage.get_harmony(id) {
            storage.set_harmony(id, (current + eval.harmony_impact).clamp(0.0, 1.0));
        }
    }
}

// ==================== ENTITY STORAGE (Component-Based) + Interest Management Notes ====================

pub type EntityId = u64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityType {
    pub entity_type: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionComponent {
    pub position: Position,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonyComponent {
    pub harmony: f64,
}

/// Component-based entity storage.
///
/// Prepared for future Interest Management:
/// - Entities can be efficiently queried by position + components.
/// - Interest sets can filter on specific components (e.g., only entities with low harmony).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityStorage {
    pub entity_types: HashMap<EntityId, EntityType>,
    pub positions: HashMap<EntityId, PositionComponent>,
    pub harmonies: HashMap<EntityId, HarmonyComponent>,
    next_id: EntityId,
}

impl EntityStorage {
    pub fn new() -> Self {
        Self {
            entity_types: HashMap::new(),
            positions: HashMap::new(),
            harmonies: HashMap::new(),
            next_id: 1000,
        }
    }

    pub fn next_id(&mut self) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn spawn(&mut self, entity_type: &str, position: Position, harmony: f64) -> EntityId {
        let id = self.next_id();
        self.entity_types.insert(id, EntityType { entity_type: entity_type.to_string() });
        self.positions.insert(id, PositionComponent { position });
        self.harmonies.insert(id, HarmonyComponent { harmony });
        id
    }

    pub fn get_position(&self, id: EntityId) -> Option<Position> {
        self.positions.get(&id).map(|p| p.position)
    }

    pub fn get_harmony(&self, id: EntityId) -> Option<f64> {
        self.harmonies.get(&id).map(|h| h.harmony)
    }

    pub fn set_position(&mut self, id: EntityId, position: Position) {
        if let Some(comp) = self.positions.get_mut(&id) {
            comp.position = position;
        }
    }

    pub fn set_harmony(&mut self, id: EntityId, harmony: f64) {
        if let Some(comp) = self.harmonies.get_mut(&id) {
            comp.harmony = harmony;
        }
    }

    pub fn remove(&mut self, id: EntityId) {
        self.entity_types.remove(&id);
        self.positions.remove(&id);
        self.harmonies.remove(&id);
    }

    pub fn len(&self) -> usize {
        self.entity_types.len()
    }

    // Future Interest Management hook:
    // pub fn query_in_range(&self, center: Position, radius: f32) -> Vec<EntityId> { ... }
}

// ==================== SIMULATION COMMAND SYSTEM ====================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulationCommand {
    MovePlayer { dx: f32, dy: f32 },
    MoveEntity { entity_id: EntityId, dx: f32, dy: f32 },
    TradeWithNpc { npc_index: usize, item: String, quantity: u32, sell_to_npc: bool },
    CraftItem { recipe_name: String },
    HarvestResource { coord: (i32, i32), resource: String, amount: f64 },
    ClaimChunk { coord: (i32, i32), owner_id: u64 },
    UpdateChunkResources { coord: (i32, i32), resource: String, delta: f64 },
    SpawnNpc { position: Position, npc_type: String },
    DespawnEntity { entity_id: EntityId },
    ApplyBlessing { target_entity: Option<EntityId>, amount: f64 },
    UpdateFactionStanding { faction: String, delta: f64 },
    Custom { data: String },
}

// ==================== SHARD-AWARE WORLD SIMULATION ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub harmony: f64,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            mercy: 0.82,
            harmony: 0.75,
        }
    }
}

/// Shard-aware World Simulation.
/// Each instance represents one zone/shard in the larger world.
pub struct WorldSimulation {
    pub shard_context: ShardContext,
    pub entities: EntityStorage,
    pub player: PlayerState,
    pub tick_count: u64,
}

impl WorldSimulation {
    pub fn new(shard_id: ShardId, name: &str) -> Self {
        Self {
            shard_context: ShardContext::new(shard_id, name),
            entities: EntityStorage::new(),
            player: PlayerState::default(),
            tick_count: 0,
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.authoritative_tick(dt);
    }

    pub fn authoritative_tick(&mut self, _dt: f32) {
        self.tick_count += 1;
    }

    pub fn evaluate_command_with_mercy(&mut self, command: &SimulationCommand, target_entity: Option<EntityId>) {
        let evaluation = MercyEvaluationSystem::evaluate_command(command);

        if let Some(id) = target_entity {
            MercyEvaluationSystem::apply_to_entity(&mut self.entities, id, &evaluation);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn shard_aware_simulation_works() {
        let mut world = WorldSimulation::new(1, "Sanctuary Zone");
        world.authoritative_tick(0.016);
        assert_eq!(world.shard_context.shard_id, 1);
        assert!(world.tick_count >= 1);
    }
}