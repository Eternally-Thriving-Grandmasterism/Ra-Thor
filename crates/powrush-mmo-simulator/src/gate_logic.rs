//! Gate Logic - Core simulation behavior driven by the 7 Living Mercy Gates + Geometry Resonance
//!
//! Includes synergies, diminishing returns, negative effects, and telemetry integration.

use crate::mercy_geometry::MercyGeometryEvaluation;
use tracing::debug;

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

#[derive(Debug, Clone)]
pub struct GateDebugInfo {
    pub raw_abundance: f32,
    pub diminished_abundance: f32,
    pub raw_mercy: f32,
    pub diminished_mercy: f32,
    pub raw_love: f32,
    pub diminished_love: f32,
    pub raw_truth: f32,
    pub diminished_truth: f32,
    pub raw_harmony: f32,
    pub diminished_harmony: f32,
    pub raw_joy: f32,
    pub diminished_joy: f32,
    pub raw_geometry: f32,
    pub diminished_geometry: f32,
    pub applied_synergies: f32,
    pub final_multipliers: GateEffects,
}

fn apply_diminishing_returns(value: f32, strength: f32) -> f32 {
    strength * (1.0 - (1.0 / (1.0 + value * 4.0)))
}

pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects {
    let s = &evaluation.score;

    let raw_abundance = (s.abundance - 0.5).max(0.0);
    let raw_mercy     = (s.mercy - 0.5).max(0.0);
    let raw_love      = (s.love - 0.5).max(0.0);
    let raw_truth     = (s.truth - 0.5).max(0.0);
    let raw_harmony   = (s.harmony - 0.5).max(0.0);
    let raw_joy       = (s.joy - 0.5).max(0.0);
    let raw_geometry  = (s.geometry_resonance - 0.5).max(0.0);

    let abundance = apply_diminishing_returns(raw_abundance, 0.8);
    let mercy     = apply_diminishing_returns(raw_mercy, 0.6);
    let love      = apply_diminishing_returns(raw_love, 0.5);
    let truth     = apply_diminishing_returns(raw_truth, 0.7);
    let harmony   = apply_diminishing_returns(raw_harmony, 0.6);
    let joy       = apply_diminishing_returns(raw_joy, 0.5);
    let geometry  = apply_diminishing_returns(raw_geometry, 0.9);

    // Negative effects for extremely low gates
    let abundance_penalty = if s.abundance < 0.35 { (0.35 - s.abundance) * 1.2 } else { 0.0 };
    let mercy_penalty     = if s.mercy < 0.30 { (0.30 - s.mercy) * 1.5 } else { 0.0 };
    let harmony_penalty   = if s.harmony < 0.35 { (0.35 - s.harmony) * 1.1 } else { 0.0 };
    let truth_penalty     = if s.truth < 0.30 { (0.30 - s.truth) * 1.3 } else { 0.0 };

    let base_resource = (1.0 + abundance - abundance_penalty).max(0.6);
    let base_evolution = (1.0 + mercy - mercy_penalty).max(0.5);
    let base_cooperation = 1.0 + love;
    let base_information = (1.0 + truth - truth_penalty).max(0.55);
    let base_harmony = (1.0 + harmony - harmony_penalty).max(0.6);
    let base_morale = 1.0 + joy;
    let base_geometry = 1.0 + geometry;

    let love_harmony_synergy = if s.love > 0.85 && s.harmony > 0.85 { 0.22 } else { 0.0 };
    let truth_abundance_synergy = if s.truth > 0.80 && s.abundance > 0.80 { 0.18 } else { 0.0 };
    let mercy_joy_synergy = if s.mercy > 0.90 && s.joy > 0.85 { 0.20 } else { 0.0 };
    let geometry_harmony_synergy = if s.geometry_resonance > 0.88 && s.harmony > 0.82 { 0.16 } else { 0.0 };

    let overall_health = (s.love + s.mercy + s.truth + s.abundance + s.harmony + s.joy + s.geometry_resonance) / 7.0;
    let overall_synergy = if overall_health > 0.85 { 0.12 } else { 0.0 };

    let total_synergy = love_harmony_synergy + truth_abundance_synergy + mercy_joy_synergy + geometry_harmony_synergy + overall_synergy;

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

/// Returns detailed debug/telemetry information.
pub fn get_gate_debug_info(evaluation: &MercyGeometryEvaluation) -> GateDebugInfo {
    let s = &evaluation.score;

    let raw_abundance = (s.abundance - 0.5).max(0.0);
    let raw_mercy     = (s.mercy - 0.5).max(0.0);
    let raw_love      = (s.love - 0.5).max(0.0);
    let raw_truth     = (s.truth - 0.5).max(0.0);
    let raw_harmony   = (s.harmony - 0.5).max(0.0);
    let raw_joy       = (s.joy - 0.5).max(0.0);
    let raw_geometry  = (s.geometry_resonance - 0.5).max(0.0);

    let diminished_abundance = apply_diminishing_returns(raw_abundance, 0.8);
    let diminished_mercy     = apply_diminishing_returns(raw_mercy, 0.6);
    let diminished_love      = apply_diminishing_returns(raw_love, 0.5);
    let diminished_truth     = apply_diminishing_returns(raw_truth, 0.7);
    let diminished_harmony   = apply_diminishing_returns(raw_harmony, 0.6);
    let diminished_joy       = apply_diminishing_returns(raw_joy, 0.5);
    let diminished_geometry  = apply_diminishing_returns(raw_geometry, 0.9);

    let effects = compute_gate_effects(evaluation);

    GateDebugInfo {
        raw_abundance,
        diminished_abundance,
        raw_mercy,
        diminished_mercy,
        raw_love,
        diminished_love,
        raw_truth,
        diminished_truth,
        raw_harmony,
        diminished_harmony,
        raw_joy,
        diminished_joy,
        raw_geometry,
        diminished_geometry,
        applied_synergies: 0.0,
        final_multipliers: effects,
    }
}

/// Emit structured telemetry for gate state (integrates with tracing).
pub fn emit_gate_telemetry(evaluation: &MercyGeometryEvaluation, entity_id: &str) {
    let debug = get_gate_debug_info(evaluation);
    let effects = &debug.final_multipliers;

    debug!(
        target: "powrush::gates",
        entity_id = %entity_id,
        abundance = debug.raw_abundance,
        diminished_abundance = debug.diminished_abundance,
        mercy = debug.raw_mercy,
        diminished_mercy = debug.diminished_mercy,
        love = debug.raw_love,
        harmony = debug.raw_harmony,
        joy = debug.raw_joy,
        geometry = debug.raw_geometry,
        resource_mult = effects.resource_multiplier,
        evolution_stability = effects.evolution_stability,
        all_gates_strong = evaluation.all_gates_strong,
        "Gate telemetry emitted"
    );
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

/// Full gate logic with automatic telemetry emission.
pub fn apply_full_gate_logic(evaluation: &MercyGeometryEvaluation, entity: &mut SimEntity, entity_id: &str) {
    // Emit telemetry on every significant gate evaluation
    emit_gate_telemetry(evaluation, entity_id);

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