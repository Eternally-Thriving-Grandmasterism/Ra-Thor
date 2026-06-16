//! Gate Logic - Core simulation behavior driven by the 7 Living Mercy Gates + Geometry Resonance
//!
//! Includes sophisticated synergies + diminishing returns.

use crate::mercy_geometry::MercyGeometryEvaluation;

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

/// Apply diminishing returns to a raw bonus value.
/// Prevents infinite scaling at very high gate values.
fn apply_diminishing_returns(value: f32, strength: f32) -> f32 {
    // Classic diminishing returns: bonus grows slower as value approaches 1.0
    // Formula: strength * (1.0 - (1.0 / (1.0 + value * 4.0)))
    strength * (1.0 - (1.0 / (1.0 + value * 4.0)))
}

/// Compute gate-driven effects with synergies + **diminishing returns**.
pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects {
    let s = &evaluation.score;

    // Raw individual contributions (0.0 - 1.0 range after normalization)
    let raw_abundance = (s.abundance - 0.5).max(0.0);
    let raw_mercy     = (s.mercy - 0.5).max(0.0);
    let raw_love      = (s.love - 0.5).max(0.0);
    let raw_truth     = (s.truth - 0.5).max(0.0);
    let raw_harmony   = (s.harmony - 0.5).max(0.0);
    let raw_joy       = (s.joy - 0.5).max(0.0);
    let raw_geometry  = (s.geometry_resonance - 0.5).max(0.0);

    // Apply diminishing returns to individual contributions
    let abundance = apply_diminishing_returns(raw_abundance, 0.8);
    let mercy     = apply_diminishing_returns(raw_mercy, 0.6);
    let love      = apply_diminishing_returns(raw_love, 0.5);
    let truth     = apply_diminishing_returns(raw_truth, 0.7);
    let harmony   = apply_diminishing_returns(raw_harmony, 0.6);
    let joy       = apply_diminishing_returns(raw_joy, 0.5);
    let geometry  = apply_diminishing_returns(raw_geometry, 0.9);

    // Base effects
    let base_resource = 1.0 + abundance;
    let base_evolution = 1.0 + mercy;
    let base_cooperation = 1.0 + love;
    let base_information = 1.0 + truth;
    let base_harmony = 1.0 + harmony;
    let base_morale = 1.0 + joy;
    let base_geometry = 1.0 + geometry;

    // === Sophisticated Synergies (still benefit from diminishing returns on base) ===
    let love_harmony_synergy = if s.love > 0.85 && s.harmony > 0.85 { 0.22 } else { 0.0 };
    let truth_abundance_synergy = if s.truth > 0.80 && s.abundance > 0.80 { 0.18 } else { 0.0 };
    let mercy_joy_synergy = if s.mercy > 0.90 && s.joy > 0.85 { 0.20 } else { 0.0 };
    let geometry_harmony_synergy = if s.geometry_resonance > 0.88 && s.harmony > 0.82 { 0.16 } else { 0.0 };

    let overall_health = (s.love + s.mercy + s.truth + s.abundance + s.harmony + s.joy + s.geometry_resonance) / 7.0;
    let overall_synergy = if overall_health > 0.85 { 0.12 } else { 0.0 };

    GateEffects {
        resource_multiplier: (base_resource + truth_abundance_synergy + overall_synergy).min(2.8),
        evolution_stability: (base_evolution + mercy_joy_synergy + overall_synergy).min(2.5),
        cooperation_bonus: (base_cooperation + love_harmony_synergy + overall_synergy).min(2.3),
        information_accuracy: (base_information + truth_abundance_synergy + overall_synergy).min(2.4),
        harmony_stability: (base_harmony + love_harmony_synergy + geometry_harmony_synergy + overall_synergy).min(2.5),
        morale_bonus: (base_morale + mercy_joy_synergy + overall_synergy).min(2.2),
        geometry_structural_bonus: (base_geometry + geometry_harmony_synergy + overall_synergy).min(2.9),
    }
}

// === Application helpers ===

pub fn apply_resource_generation(evaluation: &MercyGeometryEvaluation, base_amount: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_amount * effects.resource_multiplier).max(0.1)
}

pub fn apply_evolution_stability(evaluation: &MercyGeometryEvaluation, base_stability: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_stability * effects.evolution_stability).clamp(0.3, 1.8)
}

pub fn apply_cooperation_bonus(evaluation: &MercyGeometryEvaluation, base_cooperation: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_cooperation * effects.cooperation_bonus
}

pub fn apply_information_accuracy(evaluation: &MercyGeometryEvaluation, base_accuracy: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_accuracy * effects.information_accuracy).clamp(0.4, 1.6)
}

pub fn apply_geometry_structural_bonus(evaluation: &MercyGeometryEvaluation, base_stability: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_stability * effects.geometry_structural_bonus
}

pub fn apply_full_gate_logic(evaluation: &MercyGeometryEvaluation, entity: &mut SimEntity) {
    entity.resource_rate = apply_resource_generation(evaluation, entity.base_resource_rate);
    entity.evolution_stability = apply_evolution_stability(evaluation, entity.base_evolution_stability);
    entity.cooperation = apply_cooperation_bonus(evaluation, entity.base_cooperation);
    entity.information_accuracy = apply_information_accuracy(evaluation, entity.base_information_accuracy);
    entity.geometry_stability = apply_geometry_structural_bonus(evaluation, entity.base_geometry_stability);
}

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