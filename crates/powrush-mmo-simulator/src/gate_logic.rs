//! Gate Logic - Risk/reward for formal Lean verification attempts.

use crate::mercy_geometry::MercyGeometryEvaluation;
use mercy_threshold_wasm::MercyThresholdBridge;
use tracing::{debug, info, warn};

// ... (types remain)

#[derive(Debug, Clone, Default)]
pub struct GateEffects { /* ... */ }

#[derive(Debug, Clone)]
pub struct GateDebugInfo { /* ... */ }

#[derive(Debug, Clone)]
pub enum GateEvent {
    MajorSynergy { entity_id: String, synergy_type: String, message: String },
    LowGatePenalty { entity_id: String, gate: String, value: f32, message: String },
    ExceptionalGateHealth { entity_id: String, health: f32, message: String },
    FormalVerificationFailed { entity_id: String, message: String }, // NEW
}

#[derive(Debug, Clone)]
pub struct PowrushEntity { /* ... */ }

fn evaluation_to_mercy_threshold_params(evaluation: &MercyGeometryEvaluation) -> (u32, u32, bool, f64) {
    (12, 14, false, 0.999)
}

fn apply_bonus_diminishing(value: f32) -> f32 {
    if value <= 1.15 { value } else { 1.15 + ((value - 1.15) * 0.4) }
}

pub fn apply_formal_confirmation_bonus(evaluation: &MercyGeometryEvaluation, all_gates_strong: bool) -> GateEffects {
    // existing dynamic implementation
    compute_gate_effects(evaluation)
}

/// Apply a small penalty when formal verification is attempted and fails.
/// This creates meaningful risk/reward around using the formal Lean path.
pub fn apply_formal_verification_failure_penalty(
    evaluation: &MercyGeometryEvaluation,
) -> GateEffects {
    let base = compute_gate_effects(evaluation);

    // Small but noticeable penalties for failed formal attempt
    GateEffects {
        resource_multiplier: base.resource_multiplier * 0.92,
        evolution_stability: base.evolution_stability * 0.88,
        cooperation_bonus: base.cooperation_bonus * 0.95,
        information_accuracy: base.information_accuracy * 0.90,
        harmony_stability: base.harmony_stability * 0.93,
        morale_bonus: base.morale_bonus * 0.94,
        geometry_structural_bonus: base.geometry_structural_bonus * 0.91,
    }
}

pub fn example_simulation_tick_with_risk_reward(
    entities: &mut [PowrushEntity],
    evaluations: &[MercyGeometryEvaluation],
    bridge: Option<&mut MercyThresholdBridge>,
) {
    for (i, entity) in entities.iter_mut().enumerate() {
        if let Some(evaluation) = evaluations.get(i) {
            let entity_id = entity.id.clone();

            let (all_gates_strong, used_formal, formal_failed) =
                if let Some(b) = bridge.as_mut() {
                    let (v, f, c, mv) = evaluation_to_mercy_threshold_params(evaluation);

                    match b.check_all_gates_strong(v, f, c, mv) {
                        Ok(true) => (true, true, false),
                        Ok(false) => (false, true, true),   // Attempted formal check and failed
                        Err(_) => (evaluation.all_gates_strong, false, false),
                    }
                } else {
                    (evaluation.all_gates_strong, false, false)
                };

            let effects = if used_formal && all_gates_strong {
                apply_formal_confirmation_bonus(evaluation, true)
            } else if formal_failed {
                // RISK: Failed formal attempt applies penalty
                warn!("Formal verification failed for {} - applying penalty", entity_id);
                apply_formal_verification_failure_penalty(evaluation)
            } else {
                compute_gate_effects(evaluation)
            };

            // Apply effects
            entity.resource_stock = (entity.resource_stock * effects.resource_multiplier).max(1.0);
            entity.stability = (entity.stability * effects.evolution_stability).clamp(0.3, 1.8);
            entity.cooperation = (entity.cooperation * effects.cooperation_bonus).clamp(0.2, 2.0);
            entity.information_accuracy = (entity.information_accuracy * effects.information_accuracy).clamp(0.4, 1.6);
            entity.geometry_stability = (entity.geometry_stability * effects.geometry_structural_bonus).clamp(0.3, 1.9);

            // Extra evolution progress only on successful formal confirmation
            if used_formal && all_gates_strong {
                let avg = (evaluation.score.mercy + evaluation.score.joy + evaluation.score.geometry_resonance) / 3.0;
                let extra = ((avg - 0.85).max(0.0) * 2.5) as u32;
                entity.evolution_stage = entity.evolution_stage.saturating_add(1 + extra);
            }

            let _ = emit_and_collect_gate_events(evaluation, &entity_id);

            debug!("Tick for {} | formal_used={} | failed={}", entity_id, used_formal, formal_failed);
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