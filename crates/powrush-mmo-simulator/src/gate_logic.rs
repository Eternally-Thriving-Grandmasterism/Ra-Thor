//! Gate Logic - Dynamic bonus scaling for formal confirmation.

use crate::mercy_geometry::MercyGeometryEvaluation;
use mercy_threshold_wasm::MercyThresholdBridge;
use tracing::{debug, info};

// ... (types remain)

#[derive(Debug, Clone, Default)]
pub struct GateEffects { /* ... */ }

#[derive(Debug, Clone)]
pub struct GateDebugInfo { /* ... */ }

#[derive(Debug, Clone)]
pub enum GateEvent { /* ... */ }

#[derive(Debug, Clone)]
pub struct PowrushEntity { /* ... */ }

fn evaluation_to_mercy_threshold_params(evaluation: &MercyGeometryEvaluation) -> (u32, u32, bool, f64) {
    // existing implementation
    (12, 14, false, 0.999)
}

/// Dynamic formal confirmation bonus that scales with actual gate strength.
///
/// Higher gate values = proportionally larger formal confirmation bonuses.
pub fn apply_formal_confirmation_bonus(
    evaluation: &MercyGeometryEvaluation,
    all_gates_strong: bool,
) -> GateEffects {
    let base = compute_gate_effects(evaluation);

    if !all_gates_strong {
        return base;
    }

    let s = &evaluation.score;

    // Calculate dynamic scaling factor based on average gate strength
    let avg_gate = (s.love + s.mercy + s.truth + s.abundance +
                    s.harmony + s.joy + s.geometry_resonance) / 7.0;

    // Scale bonus between 1.05x and 1.25x depending on how high the gates are
    let dynamic_scale = 1.05 + ((avg_gate - 0.85).max(0.0) * 1.6).min(0.20);

    // Extra bonus for exceptional individual gates
    let mercy_bonus = if s.mercy > 0.92 { 0.08 } else { 0.0 };
    let joy_bonus = if s.joy > 0.88 { 0.06 } else { 0.0 };
    let geometry_bonus = if s.geometry_resonance > 0.90 { 0.10 } else { 0.0 };

    let total_scale = dynamic_scale + mercy_bonus + joy_bonus + geometry_bonus;

    GateEffects {
        resource_multiplier: base.resource_multiplier * total_scale,
        evolution_stability: base.evolution_stability * (total_scale + 0.03),
        cooperation_bonus: base.cooperation_bonus * total_scale,
        information_accuracy: base.information_accuracy * total_scale,
        harmony_stability: base.harmony_stability * (total_scale + 0.02),
        morale_bonus: base.morale_bonus * total_scale,
        geometry_structural_bonus: base.geometry_structural_bonus * (total_scale + 0.05),
    }
}

// ... (rest of the file with example tick using the dynamic version)

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
                // Dynamic evolution bonus based on how strong the gates were
                let avg = (evaluation.score.mercy + evaluation.score.joy + evaluation.score.geometry_resonance) / 3.0;
                let extra_stages = ((avg - 0.85).max(0.0) * 3.0) as u32;
                entity.evolution_stage = entity.evolution_stage.saturating_add(1 + extra_stages);

                info!("Dynamic formal bonus applied to {} (avg_gate={:.2})", entity_id, avg);
            }

            let _events = emit_and_collect_gate_events(evaluation, &entity_id);
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