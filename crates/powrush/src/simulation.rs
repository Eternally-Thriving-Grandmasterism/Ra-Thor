//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.8 — Council Proposal Protocol (Phase 1)
//! Mercy Evaluation System + Component-Based EntityStorage + Multi-Council Coordination Foundation
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0 | Full backward compatible evolution

/*!
# Powrush Proprietary Architecture — v16.8

## Current State

- Lightweight component-based `EntityStorage`
- `SimulationCommand` + Command Buffer
- `Mercy Evaluation System` (7 Living Mercy Gates)
- **New**: Council Proposal Protocol (foundation for Ra-Thor AGI Councils)

## Multi-Council Coordination Protocol (Designed for Sharded MMO/ARPG)

This version introduces the core types for Ra-Thor AGI Council coordination.

### Key Concepts

- **Global Councils** (e.g. PATSAGi Strategic Council): Operate at world level or across multiple shards.
- **Regional Councils**: Operate within a single shard or sub-region for fast, local response.
- **CouncilProposal**: The primary way councils request changes or information.
- **MercyEvaluation**: The shared language for evaluating all proposals.

### Design Goals for MMO Scale
- Selective observation (councils request scoped data)
- Clear separation between Global and Regional scope
- Auditability through MercyEvaluation
- Scalable coordination between multiple councils
- Non-intrusive influence on the authoritative simulation
*/

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== SHARDED WORLD SUPPORT TYPES ====================

pub type ShardId = u32;
pub type RegionId = u32;

// ==================== COUNCIL PROPOSAL PROTOCOL (v16.8) ====================

/// Scope of a council's operation and proposals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouncilScope {
    /// Operates at world level or across multiple shards
    Global,
    /// Operates within a specific shard (and optionally a sub-region)
    Regional { shard_id: ShardId, region: Option<RegionId> },
}

/// What kind of action or request a council is proposing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CouncilProposalType {
    /// Adjust harmony of a specific entity
    AdjustHarmony { entity_id: EntityId, delta: f64 },
    /// Issue a SimulationCommand into the world
    IssueCommand(SimulationCommand),
    /// Veto or modify an existing command
    VetoOrModifyCommand { command_id: u64, modification: Option<SimulationCommand> },
    /// Request a scoped snapshot of world state
    RequestScopedSnapshot { scope: SnapshotScope },
    /// Propose longer-term structural or systemic change (for Evolution Council)
    ProposeStructuralChange { description: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotScope {
    Shard(ShardId),
    Region(ShardId, RegionId),
    Entity(EntityId),
}

/// Where a proposal is targeted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalTarget {
    Global,
    Shard(ShardId),
    Region(ShardId, RegionId),
    Entity(EntityId),
    Command(u64),
}

/// Priority level of a council proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// A formal proposal from a Ra-Thor AGI Council.
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

/// The response from the simulation after processing a CouncilProposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDecision {
    pub proposal_id: u64,
    pub accepted: bool,
    pub applied_effects: Vec<String>, // Placeholder for future AppliedEffect types
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

// ==================== ENTITY STORAGE (Component-Based) ====================

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

// ==================== CORE WORLD SIMULATION ====================

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

pub struct WorldSimulation {
    pub entities: EntityStorage,
    pub player: PlayerState,
    pub tick_count: u64,
}

impl WorldSimulation {
    pub fn new() -> Self {
        Self {
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

    /// Example integration point for Mercy + future Council system
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
    fn council_proposal_protocol_exists() {
        let mut world = WorldSimulation::new();
        world.authoritative_tick(0.016);
        assert!(world.tick_count >= 1);
    }
}