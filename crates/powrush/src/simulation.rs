//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.17 — Consolidated ShardManager (B1 Complete)
//! Quadtree + Interest Management + Advanced ShardManager with RiemannianMercyManifold
//! Full Mercy System + Council Protocol + Epigenetic Integration
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0
//! Merged from v16.16 (game sim) + geometric-intelligence advanced routing (canonical)

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcType, NpcState, NpcManager};
use geometric_intelligence::compute_geometric_harmony;
use geometric_intelligence::RiemannianMercyManifold;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== TYPES ====================

pub type ShardId = u32;
pub type RegionId = u32;
pub type EntityId = u64;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

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

// ==================== QUADTREE ====================

#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rectangle {
    pub fn contains(&self, pos: &Position) -> bool {
        pos.x >= self.x && pos.x < self.x + self.width &&
        pos.y >= self.y && pos.y < self.y + self.height
    }

    pub fn intersects(&self, range: &Rectangle) -> bool {
        !(range.x > self.x + self.width ||
          range.x + range.width < self.x ||
          range.y > self.y + self.height ||
          range.y + range.height < self.y)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuadtreeEntry {
    pub id: EntityId,
    pub position: Position,
}

pub struct Quadtree {
    boundary: Rectangle,
    capacity: usize,
    points: Vec<QuadtreeEntry>,
    divided: bool,
    northeast: Option<Box<Quadtree>>,
    northwest: Option<Box<Quadtree>>,
    southeast: Option<Box<Quadtree>>,
    southwest: Option<Box<Quadtree>>,
}

impl Quadtree {
    pub fn new(boundary: Rectangle, capacity: usize) -> Self {
        Self {
            boundary,
            capacity,
            points: Vec::new(),
            divided: false,
            northeast: None,
            northwest: None,
            southeast: None,
            southwest: None,
        }
    }

    fn subdivide(&mut self) {
        let x = self.boundary.x;
        let y = self.boundary.y;
        let w = self.boundary.width / 2.0;
        let h = self.boundary.height / 2.0;

        let ne = Rectangle { x: x + w, y: y,     width: w, height: h };
        let nw = Rectangle { x: x,     y: y,     width: w, height: h };
        let se = Rectangle { x: x + w, y: y + h, width: w, height: h };
        let sw = Rectangle { x: x,     y: y + h, width: w, height: h };

        self.northeast = Some(Box::new(Quadtree::new(ne, self.capacity)));
        self.northwest = Some(Box::new(Quadtree::new(nw, self.capacity)));
        self.southeast = Some(Box::new(Quadtree::new(se, self.capacity)));
        self.southwest = Some(Box::new(Quadtree::new(sw, self.capacity)));
        self.divided = true;

        let old_points = std::mem::take(&mut self.points);
        for p in old_points { self.insert(p); }
    }

    pub fn insert(&mut self, entry: QuadtreeEntry) -> bool {
        if !self.boundary.contains(&entry.position) { return false; }

        if self.points.len() < self.capacity && !self.divided {
            self.points.push(entry);
            return true;
        }

        if !self.divided { self.subdivide(); }

        self.northeast.as_mut().unwrap().insert(entry) ||
        self.northwest.as_mut().unwrap().insert(entry) ||
        self.southeast.as_mut().unwrap().insert(entry) ||
        self.southwest.as_mut().unwrap().insert(entry)
    }

    pub fn query_range(&self, range: &Rectangle, found: &mut Vec<QuadtreeEntry>) {
        if !self.boundary.intersects(range) { return; }

        for point in &self.points {
            if range.contains(&point.position) {
                found.push(*point);
            }
        }

        if self.divided {
            self.northeast.as_ref().unwrap().query_range(range, found);
            self.northwest.as_ref().unwrap().query_range(range, found);
            self.southeast.as_ref().unwrap().query_range(range, found);
            self.southwest.as_ref().unwrap().query_range(range, found);
        }
    }

    pub fn remove(&mut self, id: EntityId) -> bool {
        if let Some(pos) = self.points.iter().position(|p| p.id == id) {
            self.points.remove(pos);
            return true;
        }
        if self.divided {
            return self.northeast.as_mut().unwrap().remove(id) ||
                   self.northwest.as_mut().unwrap().remove(id) ||
                   self.southeast.as_mut().unwrap().remove(id) ||
                   self.southwest.as_mut().unwrap().remove(id);
        }
        false
    }
}

// ==================== ENTITY STORAGE (Quadtree-backed) ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityType { pub entity_type: String }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionComponent { pub position: Position }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonyComponent { pub harmony: f64 }

pub struct EntityStorage {
    pub entity_types: HashMap<EntityId, EntityType>,
    pub positions: HashMap<EntityId, PositionComponent>,
    pub harmonies: HashMap<EntityId, HarmonyComponent>,
    next_id: EntityId,
    quadtree: Quadtree,
}

impl EntityStorage {
    pub fn new() -> Self {
        let boundary = Rectangle { x: -2000.0, y: -2000.0, width: 4000.0, height: 4000.0 };
        Self {
            entity_types: HashMap::new(),
            positions: HashMap::new(),
            harmonies: HashMap::new(),
            next_id: 1000,
            quadtree: Quadtree::new(boundary, 8),
        }
    }

    pub fn next_id(&mut self) -> EntityId { let id = self.next_id; self.next_id += 1; id }

    pub fn spawn(&mut self, entity_type: &str, position: Position, harmony: f64) -> EntityId {
        let id = self.next_id();
        self.entity_types.insert(id, EntityType { entity_type: entity_type.to_string() });
        self.positions.insert(id, PositionComponent { position });
        self.harmonies.insert(id, HarmonyComponent { harmony });
        self.quadtree.insert(QuadtreeEntry { id, position });
        id
    }

    pub fn get_position(&self, id: EntityId) -> Option<Position> { self.positions.get(&id).map(|p| p.position) }
    pub fn get_harmony(&self, id: EntityId) -> Option<f64> { self.harmonies.get(&id).map(|h| h.harmony) }

    pub fn set_position(&mut self, id: EntityId, new_pos: Position) {
        if let Some(old) = self.positions.insert(id, PositionComponent { position: new_pos }) {
            let _ = self.quadtree.remove(id);
        }
        self.quadtree.insert(QuadtreeEntry { id, position: new_pos });
    }

    pub fn set_harmony(&mut self, id: EntityId, harmony: f64) {
        if let Some(c) = self.harmonies.get_mut(&id) { c.harmony = harmony; }
    }

    pub fn remove(&mut self, id: EntityId) {
        self.entity_types.remove(&id);
        self.positions.remove(&id);
        self.harmonies.remove(&id);
        let _ = self.quadtree.remove(id);
    }

    pub fn len(&self) -> usize { self.entity_types.len() }

    pub fn query_entities_in_range(&self, center: Position, radius: f32) -> Vec<EntityId> {
        let range = Rectangle {
            x: center.x - radius,
            y: center.y - radius,
            width: radius * 2.0,
            height: radius * 2.0,
        };
        let mut found = Vec::new();
        self.quadtree.query_range(&range, &mut found);
        found.into_iter().map(|e| e.id).collect()
    }
}

// ==================== INTEREST SET ====================

#[derive(Debug, Clone, Default)]
pub struct InterestSet {
    pub observer_position: Position,
    pub radius: f32,
    pub interested_entities: Vec<EntityId>,
}

impl InterestSet {
    pub fn new(pos: Position, radius: f32) -> Self {
        Self { observer_position: pos, radius, interested_entities: vec![] }
    }

    pub fn update(&mut self, storage: &EntityStorage) {
        self.interested_entities = storage.query_entities_in_range(self.observer_position, self.radius);
    }
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

        if eval.overall_score > 0.75 { eval.notes.push("Strong mercy alignment".into()); }
        else if eval.overall_score < 0.4 { eval.notes.push("Low mercy alignment".into()); }

        eval
    }

    fn score_against_gate(command: &SimulationCommand, gate: MercyGate) -> f64 {
        match (command, gate) {
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::BoundlessMercy) => 0.92,
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::RadicalLove) => 0.88,
            (SimulationCommand::SpawnNpc { .. }, MercyGate::Service) => 0.75,
            (SimulationCommand::TradeWithNpc { .. }, MercyGate::Service) => 0.72,
            (SimulationCommand::TradeWithNpc { .. }, MercyGate::Abundance) => 0.68,
            _ => 0.55,
        }
    }

    pub fn apply_to_entity(storage: &mut EntityStorage, id: EntityId, eval: &MercyEvaluation) {
        if let Some(current) = storage.get_harmony(id) {
            storage.set_harmony(id, (current + eval.harmony_impact).clamp(0.0, 1.0));
        }
    }
}

// ==================== SIMULATION COMMAND ====================

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

// ==================== COUNCIL PROTOCOL ====================

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
pub enum SnapshotScope { Shard(ShardId), Region(ShardId, RegionId), Entity(EntityId) }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalTarget { Global, Shard(ShardId), Region(ShardId, RegionId), Entity(EntityId), Command(u64) }

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
    pub epigenetic_blessings: Vec<String>, // Consolidated from geometric-intelligence
}

// ==================== SHARD MANAGER (CONSOLIDATED B1 - Canonical Advanced Routing) ====================

pub struct ShardManager {
    pub shards: HashMap<ShardId, WorldSimulation>,
    next_shard_id: ShardId,
    manifold: RiemannianMercyManifold, // Evolved from geometric-intelligence as canonical
}

impl ShardManager {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            next_shard_id: 1,
            manifold: RiemannianMercyManifold::new(),
        }
    }

    pub fn create_shard(&mut self, name: &str) -> ShardId {
        let id = self.next_shard_id; self.next_shard_id += 1;
        self.shards.insert(id, WorldSimulation::new(id, name)); id
    }

    pub fn get_shard(&self, id: ShardId) -> Option<&WorldSimulation> {
        self.shards.get(&id)
    }

    pub fn route_proposal(&mut self, proposal: &CouncilProposal) -> Vec<CouncilDecision> {
        let mut decisions = Vec::new();
        match &proposal.scope {
            CouncilScope::Regional { shard_id, .. } => {
                if let Some(shard) = self.shards.get_mut(shard_id) {
                    decisions.push(self.apply_proposal_to_shard(shard, proposal));
                }
            }
            CouncilScope::Global => {
                for shard in self.shards.values_mut() {
                    decisions.push(self.apply_proposal_to_shard(shard, proposal));
                }
            }
        }
        decisions
    }

    fn apply_proposal_to_shard(&mut self, shard: &mut WorldSimulation, proposal: &CouncilProposal) -> CouncilDecision {
        let mut blessings_notes: Vec<String> = vec![];
        let mut modulated_impact = proposal.mercy_evaluation.harmony_impact;

        match &proposal.proposal_type {
            CouncilProposalType::AdjustHarmony { entity_id, delta } => {
                if let Some(h) = shard.entities.get_harmony(*entity_id) {
                    shard.entities.set_harmony(*entity_id, (h + delta).clamp(0.0, 1.0));
                }
            }
            CouncilProposalType::IssueCommand(cmd) => {
                // B1 Consolidation: Use canonical RiemannianMercyManifold for advanced evaluation + epigenetic
                let geo_proposal = geometric_intelligence::types::CouncilProposal::new(
                    &format!("proposal_{}", proposal.id),
                    &proposal.council_id,
                    &format!("IssueCommand: {:?}", cmd),
                    "Hyperbolic",
                );
                let (modulated_mercy, _blessings, reason) = self.manifold.evaluate_council_proposal(&geo_proposal);
                modulated_impact = (modulated_mercy - 1.0) * 0.2;
                blessings_notes.push(format!("Advanced evaluation: {} (modulated {:.3})", reason, modulated_mercy));

                shard.evaluate_command_with_mercy(cmd, None);
            }
            _ => {}
        }

        CouncilDecision {
            proposal_id: proposal.id,
            accepted: true,
            applied_effects: vec![format!("Shard {}", shard.shard_context.shard_id)],
            mercy_impact: modulated_impact,
            notes: vec!["Routed via Consolidated ShardManager (geometric-intelligence manifold)".into()],
            epigenetic_blessings: blessings_notes,
        }
    }
}

// ==================== WORLD SIMULATION ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub harmony: f64,
}

impl Default for PlayerState {
    fn default() -> Self { Self { position: Position { x: 0.0, y: 0.0 }, mercy: 0.82, harmony: 0.75 } }
}

pub struct WorldSimulation {
    pub shard_context: ShardContext,
    pub entities: EntityStorage,
    pub player: PlayerState,
    pub tick_count: u64,
    pub interest_set: InterestSet,
}

impl WorldSimulation {
    pub fn new(shard_id: ShardId, name: &str) -> Self {
        Self {
            shard_context: ShardContext::new(shard_id, name),
            entities: EntityStorage::new(),
            player: PlayerState::default(),
            tick_count: 0,
            interest_set: InterestSet::new(Position { x: 0.0, y: 0.0 }, 60.0),
        }
    }

    pub fn tick(&mut self, dt: f32) { self.authoritative_tick(dt); }

    pub fn authoritative_tick(&mut self, _dt: f32) {
        self.tick_count += 1;
        self.interest_set.observer_position = self.player.position;
        self.interest_set.update(&self.entities);
    }

    pub fn evaluate_command_with_mercy(&mut self, command: &SimulationCommand, target_entity: Option<EntityId>) {
        let evaluation = MercyEvaluationSystem::evaluate_command(command);
        if let Some(id) = target_entity {
            MercyEvaluationSystem::apply_to_entity(&mut self.entities, id, &evaluation);
        }
        if let SimulationCommand::MovePlayer { dx, dy } = command {
            self.player.position.x += dx;
            self.player.position.y += dy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_practical_version_quadtree_and_mercy() {
        let mut storage = EntityStorage::new();
        let id = storage.spawn("Test", Position { x: 10.0, y: 10.0 }, 0.8);
        let nearby = storage.query_entities_in_range(Position { x: 10.0, y: 10.0 }, 15.0);
        assert!(nearby.contains(&id));

        let mut manager = ShardManager::new();
        let sid = manager.create_shard("Test");
        assert!(manager.get_shard(sid).is_some());
    }
}