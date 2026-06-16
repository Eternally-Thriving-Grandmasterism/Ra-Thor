//! Gate Logic - Core simulation behavior driven by the 7 Living Mercy Gates + Geometry Resonance
//!
//! This module contains the actual *effects* of gate scores on the Powrush-MMO world.
//! It consumes MercyGeometryEvaluation (which can come from formal Lean or native MWPO).

use crate::mercy_geometry::MercyGeometryEvaluation;

/// Effects applied to a particle / entity based on gate state
#[derive(Debug, Clone, Default)]
pub struct GateEffects {
    pub resource_multiplier: f32,
    pub evolution_stability: f32,
    pub cooperation_bonus: f32,
    pub information_accuracy: f32,
    pub harmony_stability: f32,
    pub morale_bonus: f32,
    pub geometry_structural_bonus: f32,
}

/// Compute gate-driven effects for a particle/entity
pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects {
    let score = &evaluation.score;

    GateEffects {
        resource_multiplier: 1.0 + (score.abundance - 0.5) * 0.8,           // Abundance drives resources
        evolution_stability: 1.0 + (score.mercy - 0.5) * 0.6,               // Mercy drives stable evolution
        cooperation_bonus: 1.0 + (score.love - 0.5) * 0.5,                  // Love drives cooperation
        information_accuracy: 1.0 + (score.truth - 0.5) * 0.7,              // Truth drives accurate information
        harmony_stability: 1.0 + (score.harmony - 0.5) * 0.6,               // Harmony reduces conflict
        morale_bonus: 1.0 + (score.joy - 0.5) * 0.5,                        // Joy boosts morale/productivity
        geometry_structural_bonus: 1.0 + (score.geometry_resonance - 0.5) * 0.9, // Geometry affects world structure
    }
}

/// Apply gate effects to resource generation
pub fn apply_resource_generation(evaluation: &MercyGeometryEvaluation, base_amount: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_amount * effects.resource_multiplier).max(0.1)
}

/// Apply gate effects to evolution / mutation stability
pub fn apply_evolution_stability(evaluation: &MercyGeometryEvaluation, base_stability: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_stability * effects.evolution_stability).clamp(0.3, 1.8)
}

/// Apply gate effects to faction / group cooperation
pub fn apply_cooperation_bonus(evaluation: &MercyGeometryEvaluation, base_cooperation: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_cooperation * effects.cooperation_bonus
}

/// Apply gate effects to information / detection accuracy
pub fn apply_information_accuracy(evaluation: &MercyGeometryEvaluation, base_accuracy: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_accuracy * effects.information_accuracy).clamp(0.4, 1.6)
}

/// Apply gate effects to world geometry / structure stability
pub fn apply_geometry_structural_bonus(evaluation: &MercyGeometryEvaluation, base_stability: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_stability * effects.geometry_structural_bonus
}

/// Full gate-driven update for a simulation entity (example integration point)
pub fn apply_full_gate_logic(evaluation: &MercyGeometryEvaluation, entity: &mut SimEntity) {
    entity.resource_rate = apply_resource_generation(evaluation, entity.base_resource_rate);
    entity.evolution_stability = apply_evolution_stability(evaluation, entity.base_evolution_stability);
    entity.cooperation = apply_cooperation_bonus(evaluation, entity.base_cooperation);
    entity.information_accuracy = apply_information_accuracy(evaluation, entity.base_information_accuracy);
    entity.geometry_stability = apply_geometry_structural_bonus(evaluation, entity.base_geometry_stability);
}

/// Placeholder for a simulation entity (replace with real Powrush entity later)
#[derive(Debug, Clone)]
pub struct SimEntity {
    pub base_resource_rate: f32,
    pub resource_rate: f32,
    pub base_evolution_stability: f32,
    pub evolution_stability: f32,
    pub base_cooperation: f32,
    pub cooperation: f32,
    pub base_information_accuracy: f32,
    pub information_accuracy: f32,
    pub base_geometry_stability: f32,
    pub geometry_stability: f32,
}

impl Default for SimEntity {
    fn default() -> Self {
        Self {
            base_resource_rate: 1.0,
            resource_rate: 1.0,
            base_evolution_stability: 1.0,
            evolution_stability: 1.0,
            base_cooperation: 1.0,
            cooperation: 1.0,
            base_information_accuracy: 1.0,
            information_accuracy: 1.0,
            base_geometry_stability: 1.0,
            geometry_stability: 1.0,
        }
    }
}