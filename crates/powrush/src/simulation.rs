//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.10 — Proposal Routing + ShardManager + Interest Management Foundations
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0 | Full backward compatible evolution

/*!
# Powrush v16.10 — Advanced Sharded Architecture Foundations

This version advances three key areas:

1. **Proposal Routing** based on `CouncilScope` (Regional vs Global)
2. **Basic ShardManager** for coordinating multiple zone simulations
3. **Interest Management** scaffolding on component-based `EntityStorage`

See `docs/powrush/sharding-architecture.md` for the overarching design.
*/

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== SHARD & COUNCIL TYPES ====================

pub type ShardId = u32;
pub type RegionId = u32;

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
pub enum ProposalPriority {
    Low, Normal, High, Critical,
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
pub enum MercyGate { /* ... same as before ... */ }

impl MercyGate { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyEvaluation { /* ... */ }

impl MercyEvaluation { /* ... */ }

pub struct MercyEvaluationSystem;

impl MercyEvaluationSystem {
    pub fn evaluate_command(command: &SimulationCommand) -> MercyEvaluation { /* ... */ }
    fn score_against_gate(command: &SimulationCommand, gate: MercyGate) -> f64 { /* ... */ }
    pub fn apply_to_entity(storage: &mut EntityStorage, id: EntityId, eval: &MercyEvaluation) { /* ... */ }
}

// ==================== ENTITY STORAGE + INTEREST MANAGEMENT ====================

pub type EntityId = u64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityType { pub entity_type: String }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionComponent { pub position: Position }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonyComponent { pub harmony: f64 }

/// Component-based storage prepared for Interest Management.
pub struct EntityStorage {
    pub entity_types: HashMap<EntityId, EntityType>,
    pub positions: HashMap<EntityId, PositionComponent>,
    pub harmonies: HashMap<EntityId, HarmonyComponent>,
    next_id: EntityId,
}

impl EntityStorage {
    pub fn new() -> Self { /* ... */ }
    pub fn next_id(&mut self) -> EntityId { /* ... */ }
    pub fn spawn(&mut self, entity_type: &str, position: Position, harmony: f64) -> EntityId { /* ... */ }
    pub fn get_position(&self, id: EntityId) -> Option<Position> { /* ... */ }
    pub fn get_harmony(&self, id: EntityId) -> Option<f64> { /* ... */ }
    pub fn set_position(&mut self, id: EntityId, position: Position) { /* ... */ }
    pub fn set_harmony(&mut self, id: EntityId, harmony: f64) { /* ... */ }
    pub fn remove(&mut self, id: EntityId) { /* ... */ }
    pub fn len(&self) -> usize { self.entity_types.len() }

    // === Interest Management Hooks (Future) ===
    /// Returns entities within a radius. To be optimized with spatial partitioning.
    pub fn query_entities_in_range(&self, _center: Position, _radius: f32) -> Vec<EntityId> {
        // Placeholder - real implementation would use spatial index
        self.positions.keys().cloned().collect()
    }
}

// ==================== SIMULATION COMMAND ====================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulationCommand { /* ... full enum ... */ }

// ==================== SHARD MANAGER + PROPOSAL ROUTING ====================

/// Manages multiple zone/shard simulations.
pub struct ShardManager {
    pub shards: HashMap<ShardId, WorldSimulation>,
    next_shard_id: ShardId,
}

impl ShardManager {
    pub fn new() -> Self {
        Self { shards: HashMap::new(), next_shard_id: 1 }
    }

    pub fn create_shard(&mut self, name: &str) -> ShardId {
        let id = self.next_shard_id;
        self.next_shard_id += 1;
        self.shards.insert(id, WorldSimulation::new(id, name));
        id
    }

    pub fn get_shard(&self, shard_id: ShardId) -> Option<&WorldSimulation> {
        self.shards.get(&shard_id)
    }

    pub fn get_shard_mut(&mut self, shard_id: ShardId) -> Option<&mut WorldSimulation> {
        self.shards.get_mut(&shard_id)
    }

    /// Routes a CouncilProposal to the appropriate simulation(s) based on scope.
    pub fn route_proposal(&mut self, proposal: &CouncilProposal) -> Vec<CouncilDecision> {
        let mut decisions = Vec::new();

        match &proposal.scope {
            CouncilScope::Regional { shard_id, .. } => {
                if let Some(shard) = self.shards.get_mut(shard_id) {
                    let decision = self.apply_proposal_to_shard(shard, proposal);
                    decisions.push(decision);
                }
            }
            CouncilScope::Global => {
                // Global proposals may affect multiple shards or require aggregation
                for shard in self.shards.values_mut() {
                    let decision = self.apply_proposal_to_shard(shard, proposal);
                    decisions.push(decision);
                }
            }
        }

        decisions
    }

    fn apply_proposal_to_shard(&mut self, shard: &mut WorldSimulation, proposal: &CouncilProposal) -> CouncilDecision {
        // Basic handling - can be greatly expanded
        match &proposal.proposal_type {
            CouncilProposalType::AdjustHarmony { entity_id, delta } => {
                if let Some(current) = shard.entities.get_harmony(*entity_id) {
                    shard.entities.set_harmony(*entity_id, (current + delta).clamp(0.0, 1.0));
                }
            }
            CouncilProposalType::IssueCommand(cmd) => {
                // In real system, this would go through command buffer + mercy evaluation
                shard.evaluate_command_with_mercy(cmd, None);
            }
            _ => {}
        }

        CouncilDecision {
            proposal_id: proposal.id,
            accepted: true,
            applied_effects: vec![format!("Processed by shard {}", shard.shard_context.shard_id)],
            mercy_impact: proposal.mercy_evaluation.harmony_impact,
            notes: vec!["Routed via ShardManager".to_string()],
        }
    }
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
}

impl WorldSimulation {
    pub fn new(shard_id: ShardId, name: &str) -> Self { /* ... */ }
    pub fn tick(&mut self, dt: f32) { self.authoritative_tick(dt); }
    pub fn authoritative_tick(&mut self, _dt: f32) { self.tick_count += 1; }
    pub fn evaluate_command_with_mercy(&mut self, command: &SimulationCommand, target_entity: Option<EntityId>) { /* ... */ }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_manager_and_proposal_routing() {
        let mut manager = ShardManager::new();
        let shard_id = manager.create_shard("Test Zone");

        let proposal = CouncilProposal {
            id: 1,
            council_id: "TestCouncil".to_string(),
            scope: CouncilScope::Regional { shard_id, region: None },
            proposal_type: CouncilProposalType::AdjustHarmony { entity_id: 100, delta: 0.1 },
            target: ProposalTarget::Entity(100),
            mercy_evaluation: MercyEvaluation::new(),
            priority: ProposalPriority::Normal,
            reasoning: "Test proposal".to_string(),
            timestamp: 0,
        };

        let decisions = manager.route_proposal(&proposal);
        assert!(!decisions.is_empty());
    }
}