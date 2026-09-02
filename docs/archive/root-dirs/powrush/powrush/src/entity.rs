//! Sovereign Entity System for Powrush Living Simulation
//!
//! Defines Human / AI / AGI entities that coexist, learn skills, contribute to RBE,
//! and maintain shared valence/mercy fields. Directly extends the multi-agent
//! orchestration blueprint and integrates with CliffordHealingField,
//! ServerUnlockState, PATSAGi councils, and Hyperon/Metta reasoning.
//!
//! Every entity is mercy-gated. Contributions drive universal thriving dividends.
//!
//! RBE CONTRIBUTION MECHANICS (investigated & implemented):
//! - Contribution is recorded via `contribute(skill, amount)`
//! - Personal thriving dividend = f(contribution_ratio, valence, global_mercy_flow, council_alignment)
//! - Dividend is applied back as skill growth + valence boost + small contribution feedback
//! - Global distribution pass uses current lattice state (healing field + PATSAGi progress)
//! - Non-bypassable mercy gates at every step. Post-scarcity loop: contribute → receive → grow → contribute more.
//! - Designed for extension into full resource access rights, faction pools, and
//!   hyperon_metta_pln reasoned fairness proofs.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::Component;

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
/// Now a proper Bevy Component so it can be queried with ECS Query syntax.
#[derive(Debug, Clone, Component)]
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

    /// Calculate this entity's share of the universal thriving dividend.
    /// total_system_contributions: aggregate across all sovereign entities
    /// global_mercy_flow: current mercy_flow from CliffordHealingField (0.0-1.0)
    /// council_alignment: PATSAGi / unlock progress proxy (0.0-1.0)
    pub fn calculate_personal_thriving_dividend(
        &self,
        total_system_contributions: u64,
        global_mercy_flow: f32,
        council_alignment: f32,
    ) -> u64 {
        if total_system_contributions == 0 {
            return 50; // base thriving floor for new or low-activity entities
        }

        let contribution_ratio =
            self.contributions as f64 / total_system_contributions as f64;
        let base_share = contribution_ratio * 1000.0; // demo economic scale

        let valence_bonus =
            (self.valence as f64 * 250.0) * (council_alignment as f64);

        let mercy_multiplier =
            (global_mercy_flow as f64 * 0.8 + 0.2).clamp(0.5, 1.5);

        let raw_dividend = (base_share + valence_bonus) * mercy_multiplier;

        // Final mercy gate: guaranteed floor + soft upper bound
        raw_dividend.clamp(10.0, 5000.0) as u64
    }

    /// Apply a received thriving dividend back into the entity.
    /// Closes the RBE positive feedback loop: contribution → dividend → growth.
    pub fn apply_thriving_dividend(&mut self, dividend: u64, mercy_influence: f32) {
        if !self.skills.is_empty() {
            for (_skill, prof) in self.skills.iter_mut() {
                *prof = (*prof + (dividend as f32 / 100.0)).min(10.0);
            }
        } else {
            self.skills
                .insert("coexistence".to_string(), 0.5 + (dividend as f32 / 500.0));
        }

        // Feedback into valence (mercy-gated)
        self.apply_valence_update(mercy_influence + (dividend as f32 / 10000.0));
        self.contributions += dividend / 5; // gentle recirculation into commons
        self.touch();
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

/// Global RBE distribution pass.
/// Call periodically from PATSAGi council systems, quantum swarm tick,
/// or dedicated orchestration. Uses live lattice state for fair,
/// mercy-aligned dividend calculation and application.
pub fn distribute_universal_thriving_dividends(
    entities: &mut [SovereignEntity],
    healing_field: &CliffordHealingField,
    unlock_state: &ServerUnlockState,
) {
    let total_contrib: u64 = entities.iter().map(|e| e.contributions).sum();
    let global_mercy = healing_field.mercy_flow as f32;
    let council_align = unlock_state.council_influence_progress;

    for entity in entities.iter_mut() {
        let dividend = entity.calculate_personal_thriving_dividend(
            total_contrib,
            global_mercy,
            council_align,
        );
        entity.apply_thriving_dividend(dividend, global_mercy);
    }
}
