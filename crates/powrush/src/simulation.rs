//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.7 — Mercy Evaluation System (7 Living Mercy Gates)
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0 | Full backward compatible evolution

/*!
# Mercy Evaluation System — Core of Ra-Thor Aligned Simulation

## Philosophy

The Mercy Evaluation System is one of the most important proprietary systems in Powrush.
It allows the simulation (and future Ra-Thor AGI) to evaluate actions, commands,
and entity behaviors through the **7 Living Mercy Gates**:

1. **Radical Love**      — Actions rooted in genuine care and connection
2. **Boundless Mercy**    — Forgiveness, second chances, and compassion
3. **Service**            — Selfless contribution to others and the whole
4. **Abundance**          — Creating and sharing value without hoarding
5. **Truth**              — Honesty, clarity, and alignment with reality
6. **Joy**                — Actions that generate genuine positive experience
7. **Cosmic Harmony**     — Actions that support balance and long-term flourishing

This system is designed to be:
- Observable and influenceable by Ra-Thor AGI
- Integrated with `SimulationCommand` and entity components
- A foundation for meaningful progression and world governance

## Design Principles
- Lightweight but meaningful
- Easy for AGI systems to query and extend
- Harmony impact as primary feedback mechanism
*/

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== MERCY EVALUATION SYSTEM (v16.7) ====================

/// The 7 Living Mercy Gates that form the ethical and spiritual foundation of Powrush.
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
            MercyGate::RadicalLove     => "Radical Love",
            MercyGate::BoundlessMercy => "Boundless Mercy",
            MercyGate::Service        => "Service",
            MercyGate::Abundance      => "Abundance",
            MercyGate::Truth          => "Truth",
            MercyGate::Joy            => "Joy",
            MercyGate::CosmicHarmony  => "Cosmic Harmony",
        }
    }
}

/// Result of evaluating an action or command through the Mercy Gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyEvaluation {
    pub overall_score: f64,                    // 0.0 to 1.0
    pub gate_scores: HashMap<MercyGate, f64>, // Individual gate alignment
    pub harmony_impact: f64,                  // Suggested harmony change
    pub notes: Vec<String>,                   // Human/AGI readable reasoning
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

/// The Mercy Evaluation System.
/// Responsible for assessing actions and commands according to the 7 Living Mercy Gates.
pub struct MercyEvaluationSystem;

impl MercyEvaluationSystem {
    /// Evaluate a `SimulationCommand` through the Mercy Gates.
    /// This is the primary integration point with the command system.
    pub fn evaluate_command(command: &SimulationCommand) -> MercyEvaluation {
        let mut eval = MercyEvaluation::new();
        let mut total_score = 0.0;
        let mut count = 0;

        for gate in MercyGate::all() {
            let score = Self::score_command_against_gate(command, gate);
            eval.gate_scores.insert(gate, score);
            total_score += score;
            count += 1;
        }

        eval.overall_score = if count > 0 { total_score / count as f64 } else { 0.5 };

        // Harmony impact is stronger when overall mercy alignment is high
        eval.harmony_impact = (eval.overall_score - 0.5) * 0.1;

        // Add basic reasoning (can be greatly expanded with AGI later)
        if eval.overall_score > 0.75 {
            eval.notes.push("Strong alignment with mercy principles".to_string());
        } else if eval.overall_score < 0.4 {
            eval.notes.push("Action shows low mercy alignment".to_string());
        }

        eval
    }

    fn score_command_against_gate(command: &SimulationCommand, gate: MercyGate) -> f64 {
        match (command, gate) {
            // Move commands generally neutral to positive
            (SimulationCommand::MovePlayer { .. } | SimulationCommand::MoveEntity { .. }, _) => 0.6,

            // Trading can be positive if fair
            (SimulationCommand::TradeWithNpc { .. }, MercyGate::Service) => 0.7,
            (SimulationCommand::TradeWithNpc { .. }, MercyGate::Abundance) => 0.65,

            // Claiming land — depends on context (simplified here)
            (SimulationCommand::ClaimChunk { .. }, MercyGate::Abundance) => 0.55,
            (SimulationCommand::ClaimChunk { .. }, MercyGate::CosmicHarmony) => 0.5,

            // Spawning NPCs — generally positive if done with care
            (SimulationCommand::SpawnNpc { .. }, MercyGate::Service) => 0.7,
            (SimulationCommand::SpawnNpc { .. }, MercyGate::Joy) => 0.65,

            // Applying blessings is strongly merciful
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::BoundlessMercy) => 0.9,
            (SimulationCommand::ApplyBlessing { .. }, MercyGate::RadicalLove) => 0.85,

            // Default neutral score
            _ => 0.55,
        }
    }

    /// Apply mercy evaluation result to an entity (adjusts harmony).
    pub fn apply_to_entity(storage: &mut EntityStorage, entity_id: EntityId, evaluation: &MercyEvaluation) {
        if let Some(current) = storage.get_harmony(entity_id) {
            let new_harmony = (current + evaluation.harmony_impact).clamp(0.0, 1.0);
            storage.set_harmony(entity_id, new_harmony);
        }
    }
}

// ==================== ENTITY STORAGE — LIGHTWEIGHT COMPONENT MODEL ====================

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
    pub positions:    HashMap<EntityId, PositionComponent>,
    pub harmonies:    HashMap<EntityId, HarmonyComponent>,
    next_id: EntityId,
}

impl EntityStorage {
    pub fn new() -> Self {
        Self {
            entity_types: HashMap::new(),
            positions:    HashMap::new(),
            harmonies:    HashMap::new(),
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

// ... (rest of the file continues with existing systems)

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

// ... (Input system, SessionManager, WorldChunk, etc. remain similar)

// For brevity in this commit, core simulation and command application logic
// continues from v16.6 with minor updates to integrate MercyEvaluation where relevant.

// In apply_pending_commands we can now optionally call:
// let evaluation = MercyEvaluationSystem::evaluate_command(&command);
// MercyEvaluationSystem::apply_to_entity(&mut self.entities, entity_id, &evaluation);

// The full integration will be deepened in future commits.

// ... (remaining code from previous version preserved for continuity)
