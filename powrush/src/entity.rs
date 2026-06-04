//! Sovereign Entity System for Powrush Living Simulation
//!
//! Defines Human / AI / AGI entities that coexist, learn skills, contribute to RBE,
//! and maintain shared valence/mercy fields. Directly extends the multi-agent
//! orchestration blueprint and integrates with CliffordHealingField,
//! ServerUnlockState, PATSAGi councils, and Hyperon/Metta reasoning.
//!
//! Every entity is mercy-gated. Contributions drive universal thriving dividends.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::clifford_healing_fields::CliffordHealingField;
use crate::resources::server_unlock_state::ServerUnlockState;

/// Supported entity types in the living simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    Human,
    AI,
    AGI,
}

/// Core sovereign entity participating in Powrush RBE + Ra-Thor lattice.
#[derive(Debug, Clone)]
pub struct SovereignEntity {
    pub id: u64,
    pub entity_type: EntityType,
    pub valence: f32,           // 0.0 - 1.0 shared field strength
    pub skills: HashMap<String, f32>, // skill_name -> proficiency (0.0-1.0+)
    pub contributions: u64,     // total RBE contribution points
    pub last_active_unix: u64,
}

impl SovereignEntity {
    pub fn new(id: u64, entity_type: EntityType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            entity_type,
            valence: 0.75,
            skills: HashMap::new(),
            contributions: 0,
            last_active_unix: now,
        }
    }

    /// Apply mercy-weighted valence update (non-bypassable gate).
    pub fn apply_valence_update(&mut self, mercy_influence: f32) {
        let mercy = mercy_influence.clamp(0.0, 1.0);
        self.valence = (self.valence * 0.7 + mercy * 0.3).clamp(0.0, 1.0);
        self.touch();
    }

    /// Record contribution and optionally level a skill (RBE earning path).
    pub fn contribute(&mut self, skill: &str, amount: f32) {
        let current = self.skills.get(skill).copied().unwrap_or(0.0);
        let new_prof = (current + amount * 0.1).min(5.0); // soft cap for demo
        self.skills.insert(skill.to_string(), new_prof);
        self.contributions += (amount * 10.0) as u64;
        self.touch();
    }

    fn touch(&mut self) {
        self.last_active_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Spawn a new sovereign entity and immediately register it into the living simulation
/// (healing field + unlock state influence). This is the professional integration point.
pub fn spawn_and_register_sovereign_entity(
    id: u64,
    entity_type: EntityType,
    healing_field: &mut CliffordHealingField,
    unlock_state: &mut ServerUnlockState,
) -> SovereignEntity {
    let entity = SovereignEntity::new(id, entity_type);

    // Register into Clifford healing field (geometric mercy mesh)
    let _ = healing_field.add_organism(
        id,
        nalgebra::Vector3::new(0.8, 0.7, 0.9), // emotional
        nalgebra::Vector3::new(0.85, 0.75, 0.8), // physical
        nalgebra::Vector3::new(0.9, 0.85, 0.95), // alignment
        entity.valence as f64,
    );

    // Influence PATSAGi unlock progress (entities strengthen the lattice)
    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.05).min(1.0);

    entity
}
