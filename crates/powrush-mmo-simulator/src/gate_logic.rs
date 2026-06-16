//! Gate Logic - Dynamic formal bonuses with diminishing returns.

use crate::mercy_geometry::MercyGeometryEvaluation;
use mercy_threshold_wasm::MercyThresholdBridge;
use tracing::{debug, info};

// ... (types)

#[derive(Debug, Clone, Default)]
pub struct GateEffects { /* ... */ }

#[derive(Debug, Clone)]
pub struct GateDebugInfo { /* ... */ }

#[derive(Debug, Clone)]
pub enum GateEvent { /* ... */ }

#[derive(Debug, Clone)]
pub struct PowrushEntity { /* ... */ }

fn evaluation_to_mercy_threshold_params(evaluation: &MercyGeometryEvaluation) -> (u32, u32, bool, f64) {
    // existing
    (12, 14, false, 0.999)
}

/// Apply diminishing returns to a bonus multiplier
fn apply_bonus_diminishing(value: f32) -> f32 {
    // Diminishing returns on very high bonuses
    if value <= 1.15 { value } else { 1.15 + ((value - 1.15) * 0.4) }
}

/// Dynamic formal confirmation bonus with diminishing returns on high values.
pub fn apply_formal_confirmation_bonus(
    evaluation: &MercyGeometryEvaluation,
    all_gates_strong: bool,
) -> GateEffects {
    let base = compute_gate_effects(evaluation);

    if !all_gates_strong {
        return base;
    }

    let s = &evaluation.score;

    let avg_gate = (s.love + s.mercy + s.truth + s.abundance +
                    s.harmony + s.joy + s.geometry_resonance) / 7.0;

    let dynamic_scale = 1.05 + ((avg_gate - 0.85).max(0.0) * 1.6).min(0.20);

    let mercy_bonus = if s.mercy > 0.92 { 0.08 } else { 0.0 };
    let joy_bonus = if s.joy > 0.88 { 0.06 } else { 0.0 };
    let geometry_bonus = if s.geometry_resonance > 0.90 { 0.10 } else { 0.0 };

    let raw_total = dynamic_scale + mercy_bonus + joy_bonus + geometry_bonus;

    // Apply diminishing returns to the final bonus
    let final_scale = apply_bonus_diminishing(raw_total);

    GateEffects {
        resource_multiplier: apply_bonus_diminishing(base.resource_multiplier * final_scale),
        evolution_stability: apply_bonus_diminishing(base.evolution_stability * (final_scale + 0.03)),
        cooperation_bonus: apply_bonus_diminishing(base.cooperation_bonus * final_scale),
        information_accuracy: apply_bonus_diminishing(base.information_accuracy * final_scale),
        harmony_stability: apply_bonus_diminishing(base.harmony_stability * (final_scale + 0.02)),
        morale_bonus: apply_bonus_diminishing(base.morale_bonus * final_scale),
        geometry_structural_bonus: apply_bonus_diminishing(base.geometry_structural_bonus * (final_scale + 0.05)),
    }
}

// ... (example tick and other functions remain)

pub fn example_simulation_tick_with_formal_reward(
    entities: &mut [PowrushEntity],
    evaluations: &[MercyGeometryEvaluation],
    bridge: Option<&mut MercyThresholdBridge>,
) {
    for (i, entity) in entities.iter_mut().enumerate() {
        if let Some(evaluation) = evaluations.get(i) {
            let entity_id = entity.id.clone();

            let (all_gates_strong, used_formal) = if let Some(b) = bridge.as_mut() {
                let (v, f, c, mv) = evaluation_to_mercy_threshold_params(evaluation);
                match b.check_all_gates_strong(v, f, c, mv) {
                    Ok(strong) => (strong, true),
                    Err(_) => (evaluation.all_gates_strong, false),
                }
            } else {
                (evaluation.all_gates_strong, false)
            };

            let effects = if used_formal {
                apply_formal_confirmation_bonus(evaluation, all_gates_strong)
            } else {
                compute_gate_effects(evaluation)
            };

            entity.resource_stock = (entity.resource_stock * effects.resource_multiplier).max(1.0);
            entity.stability = (entity.stability * effects.evolution_stability).clamp(0.3, 1.8);
            entity.cooperation = (entity.cooperation * effects.cooperation_bonus).clamp(0.2, 2.0);
            entity.information_accuracy = (entity.information_accuracy * effects.information_accuracy).clamp(0.4, 1.6);
            entity.geometry_stability = (entity.geometry_stability * effects.geometry_structural_bonus).clamp(0.3, 1.9);

            if used_formal && all_gates_strong {
                let avg = (evaluation.score.mercy + evaluation.score.joy + evaluation.score.geometry_resonance) / 3.0;
                let extra_stages = ((avg - 0.85).max(0.0) * 2.5) as u32; // slightly reduced due to diminishing
                entity.evolution_stage = entity.evolution_stage.saturating_add(1 + extra_stages);

                debug!("Dynamic formal bonus (with diminishing) applied to {}", entity_id);
            }

            let _ = emit_and_collect_gate_events(evaluation, &entity_id);
        }
    }
}

// Core functions
pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects { GateEffects::default() }
pub fn emit_and_collect_gate_events(evaluation: &MercyGeometryEvaluation, entity_id: &str) -> Vec<GateEvent> { vec![] }

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

#[derive(Debug, Clone)]
pub struct SimEntity { /* ... */ }

impl Default for SimEntity {
    fn default() -> Self { /* ... */ }
}