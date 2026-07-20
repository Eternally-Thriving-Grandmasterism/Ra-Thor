//! Gate Logic — v14.15.0
//!
//! Risk / Reward + Synergies + Diminishing Returns for the Powrush-MMO Simulator.
//! Fully mercy-gated. Integrates with MercyGeometryEvaluation + optional Lean WASM
//! formal verification via MercyThresholdBridge.
//!
//! Living Cosmic Tick aligned. Cosmic Loop is expected to be enforced by the caller.
//! Contact: info@Rathor.ai

use crate::mercy_geometry::MercyGeometryEvaluation;
use mercy_threshold_wasm::MercyThresholdBridge;
use tracing::{debug, info, warn};

// =============================================================================
// Core data structures
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct GateEffects {
    /// Multiplier for resource / abundance generation
    pub resource_multiplier: f32,
    /// Stability / evolution speed multiplier
    pub evolution_stability: f32,
    /// Cooperation / faction harmony bonus
    pub cooperation_bonus: f32,
    /// Information accuracy / truth propagation
    pub information_accuracy: f32,
    /// Harmony / structural stability
    pub harmony_stability: f32,
    /// Morale / joy influence
    pub morale_bonus: f32,
    /// Geometry / structural integrity bonus (self-healing, Daedalus-Skin style)
    pub geometry_structural_bonus: f32,
}

impl GateEffects {
    pub fn identity() -> Self {
        Self {
            resource_multiplier: 1.0,
            evolution_stability: 1.0,
            cooperation_bonus: 1.0,
            information_accuracy: 1.0,
            harmony_stability: 1.0,
            morale_bonus: 1.0,
            geometry_structural_bonus: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateDebugInfo {
    pub entity_id: String,
    pub overall_gate_health: f32,
    pub dominant_gate: String,
    pub synergy_level: f32,
    pub diminishing_applied: bool,
    pub formal_path_used: bool,
    pub formal_success: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum GateEvent {
    MajorSynergy {
        entity_id: String,
        synergy_type: String,
        message: String,
        bonus_magnitude: f32,
    },
    LowGatePenalty {
        entity_id: String,
        gate: String,
        value: f32,
        message: String,
    },
    ExceptionalGateHealth {
        entity_id: String,
        health: f32,
        message: String,
    },
    FormalVerificationFailed {
        entity_id: String,
        message: String,
    },
    FormalVerificationSuccess {
        entity_id: String,
        message: String,
    },
}

/// Primary simulation entity for Powrush-MMO gate-driven ticks.
#[derive(Debug, Clone)]
pub struct PowrushEntity {
    pub id: String,
    pub resource_stock: f32,
    pub stability: f32,
    pub cooperation: f32,
    pub information_accuracy: f32,
    pub geometry_stability: f32,
    pub evolution_stage: u32,
    pub morale: f32,
}

impl Default for PowrushEntity {
    fn default() -> Self {
        Self {
            id: "unknown".to_string(),
            resource_stock: 100.0,
            stability: 1.0,
            cooperation: 1.0,
            information_accuracy: 1.0,
            geometry_stability: 1.0,
            evolution_stage: 0,
            morale: 1.0,
        }
    }
}

// Legacy alias kept for backward compatibility
pub type SimEntity = PowrushEntity;

// =============================================================================
// Helpers
// =============================================================================

/// Soft diminishing returns — prevents infinite scaling while still rewarding excellence.
fn apply_bonus_diminishing(value: f32) -> f32 {
    if value <= 1.15 {
        value
    } else {
        1.15 + ((value - 1.15) * 0.4).min(0.6)
    }
}

// =============================================================================
// Core: compute_gate_effects
// =============================================================================

pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects {
    let s = &evaluation.score;

    // Normalize to a useful production range with mild mercy bias
    let love = (s.love as f32).clamp(0.3, 1.5);
    let mercy = (s.mercy as f32).clamp(0.3, 1.5);
    let truth = (s.truth as f32).clamp(0.3, 1.5);
    let abundance = (s.abundance as f32).clamp(0.3, 1.5);
    let harmony = (s.harmony as f32).clamp(0.3, 1.5);
    let geo = (s.geometry_resonance as f32).clamp(0.3, 1.5);
    let overall = (s.overall as f32).clamp(0.3, 1.5);

    let mut effects = GateEffects::identity();

    // Base effects from individual gates
    effects.resource_multiplier =
        1.0 + (abundance - 0.6).max(0.0) * 1.2 + (overall - 0.7).max(0.0) * 0.6;
    effects.evolution_stability = 1.0
        + (mercy - 0.55).max(0.0) * 1.4
        + if evaluation.all_gates_strong {
            0.35
        } else {
            0.0
        };
    effects.cooperation_bonus =
        1.0 + (harmony - 0.6).max(0.0) * 1.1 + (love - 0.65).max(0.0) * 0.7;
    effects.information_accuracy = 1.0 + (truth - 0.6).max(0.0) * 1.3;
    effects.harmony_stability =
        1.0 + (harmony - 0.55).max(0.0) * 1.2 + (geo - 0.6).max(0.0) * 0.8;
    effects.morale_bonus =
        1.0 + (love - 0.5).max(0.0) * 0.9 + (mercy - 0.6).max(0.0) * 0.6;
    effects.geometry_structural_bonus = 1.0 + (geo - 0.55).max(0.0) * 1.5;

    // === Synergies ===
    let mut synergy_bonus: f32 = 1.0;

    if love > 0.82 && harmony > 0.82 {
        synergy_bonus *= 1.18;
        effects.cooperation_bonus *= 1.12;
        effects.morale_bonus *= 1.10;
    }
    if truth > 0.82 && abundance > 0.82 {
        synergy_bonus *= 1.15;
        effects.information_accuracy *= 1.15;
        effects.resource_multiplier *= 1.08;
    }
    if mercy > 0.85 && overall > 0.88 {
        synergy_bonus *= 1.12;
        effects.evolution_stability *= 1.15;
        effects.morale_bonus *= 1.08;
    }
    if geo > 0.85 && harmony > 0.80 {
        synergy_bonus *= 1.10;
        effects.geometry_structural_bonus *= 1.18;
        effects.harmony_stability *= 1.10;
    }

    // Apply overall synergy lift to core multipliers
    if synergy_bonus > 1.05 {
        effects.resource_multiplier *= synergy_bonus * 0.6 + 0.4;
        effects.evolution_stability *= synergy_bonus * 0.5 + 0.5;
    }

    // === Diminishing returns ===
    effects.resource_multiplier = apply_bonus_diminishing(effects.resource_multiplier);
    effects.evolution_stability = apply_bonus_diminishing(effects.evolution_stability);
    effects.cooperation_bonus = apply_bonus_diminishing(effects.cooperation_bonus);
    effects.geometry_structural_bonus =
        apply_bonus_diminishing(effects.geometry_structural_bonus);

    // === Low-gate penalties ===
    if love < 0.45 {
        effects.morale_bonus *= 0.82;
        effects.cooperation_bonus *= 0.90;
    }
    if mercy < 0.40 {
        effects.evolution_stability *= 0.85;
        effects.morale_bonus *= 0.88;
    }
    if truth < 0.40 {
        effects.information_accuracy *= 0.80;
    }
    if harmony < 0.45 || geo < 0.45 {
        effects.harmony_stability *= 0.87;
        effects.geometry_structural_bonus *= 0.85;
    }

    // Production clamps
    effects.resource_multiplier = effects.resource_multiplier.clamp(0.4, 3.0);
    effects.evolution_stability = effects.evolution_stability.clamp(0.3, 2.8);
    effects.cooperation_bonus = effects.cooperation_bonus.clamp(0.3, 2.5);
    effects.information_accuracy = effects.information_accuracy.clamp(0.4, 2.2);
    effects.harmony_stability = effects.harmony_stability.clamp(0.4, 2.5);
    effects.morale_bonus = effects.morale_bonus.clamp(0.3, 2.5);
    effects.geometry_structural_bonus =
        effects.geometry_structural_bonus.clamp(0.4, 3.2);

    effects
}

// =============================================================================
// Event emission
// =============================================================================

pub fn emit_and_collect_gate_events(
    evaluation: &MercyGeometryEvaluation,
    entity_id: &str,
) -> Vec<GateEvent> {
    let mut events = Vec::new();
    let s = &evaluation.score;
    let overall_health = s.overall as f32;

    if s.love as f32 > 0.82 && s.harmony as f32 > 0.82 {
        events.push(GateEvent::MajorSynergy {
            entity_id: entity_id.to_string(),
            synergy_type: "Love + Harmony".to_string(),
            message: "Unity Resonance activated — cooperation and morale surging".to_string(),
            bonus_magnitude: 1.18,
        });
    }
    if s.truth as f32 > 0.82 && s.abundance as f32 > 0.82 {
        events.push(GateEvent::MajorSynergy {
            entity_id: entity_id.to_string(),
            synergy_type: "Truth + Abundance".to_string(),
            message: "Prosperity Alignment — resources and accuracy elevated".to_string(),
            bonus_magnitude: 1.15,
        });
    }
    if s.mercy as f32 > 0.85 && overall_health > 0.88 {
        events.push(GateEvent::MajorSynergy {
            entity_id: entity_id.to_string(),
            synergy_type: "Mercy + Overall".to_string(),
            message: "Blessing Cascade — evolution stability and morale lifted".to_string(),
            bonus_magnitude: 1.12,
        });
    }

    if s.love as f32 < 0.45 {
        events.push(GateEvent::LowGatePenalty {
            entity_id: entity_id.to_string(),
            gate: "Love".to_string(),
            value: s.love as f32,
            message: "Low Love detected — morale and cooperation penalized".to_string(),
        });
    }
    if s.mercy as f32 < 0.40 {
        events.push(GateEvent::LowGatePenalty {
            entity_id: entity_id.to_string(),
            gate: "Mercy".to_string(),
            value: s.mercy as f32,
            message: "Low Mercy — evolution stability reduced".to_string(),
        });
    }

    if overall_health > 0.92 && evaluation.all_gates_strong {
        events.push(GateEvent::ExceptionalGateHealth {
            entity_id: entity_id.to_string(),
            health: overall_health,
            message: "Exceptional gate health + formal confirmation — thriving maximized"
                .to_string(),
        });
    }

    events
}

// =============================================================================
// Formal verification risk / reward
// =============================================================================

fn evaluation_to_mercy_threshold_params(
    evaluation: &MercyGeometryEvaluation,
) -> (u32, u32, bool, f64) {
    let v = ((evaluation.score.geometry_resonance * 20.0) as u32)
        .max(8)
        .min(24);
    let f = ((evaluation.score.harmony * 18.0) as u32).max(6).min(22);
    let chiral = evaluation.score.geometry_resonance > 0.75;
    let mv = (evaluation.score.mercy * 1.1).clamp(0.6, 1.4);
    (v, f, chiral, mv)
}

/// Apply bonus when formal Lean verification succeeds.
pub fn apply_formal_confirmation_bonus(
    evaluation: &MercyGeometryEvaluation,
    all_gates_strong: bool,
) -> GateEffects {
    let mut base = compute_gate_effects(evaluation);
    if all_gates_strong {
        base.evolution_stability *= 1.25;
        base.resource_multiplier *= 1.12;
        base.geometry_structural_bonus *= 1.15;
        base.morale_bonus *= 1.10;
    }
    base
}

/// Apply penalty when formal verification is attempted and fails (risk realized).
pub fn apply_formal_verification_failure_penalty(
    evaluation: &MercyGeometryEvaluation,
) -> GateEffects {
    let base = compute_gate_effects(evaluation);

    GateEffects {
        resource_multiplier: (base.resource_multiplier * 0.92).max(0.5),
        evolution_stability: (base.evolution_stability * 0.88).max(0.4),
        cooperation_bonus: (base.cooperation_bonus * 0.95).max(0.5),
        information_accuracy: (base.information_accuracy * 0.90).max(0.5),
        harmony_stability: (base.harmony_stability * 0.93).max(0.5),
        morale_bonus: (base.morale_bonus * 0.94).max(0.5),
        geometry_structural_bonus: (base.geometry_structural_bonus * 0.91).max(0.5),
    }
}

// =============================================================================
// Main simulation tick
// =============================================================================

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
                        Ok(false) => (false, true, true), // Attempted + failed → risk
                        Err(e) => {
                            warn!("WASM bridge error for {}: {}", entity_id, e);
                            (evaluation.all_gates_strong, false, false)
                        }
                    }
                } else {
                    (evaluation.all_gates_strong, false, false)
                };

            let effects = if used_formal && all_gates_strong {
                info!(
                    "Formal verification SUCCESS for {} — applying confirmation bonus",
                    entity_id
                );
                apply_formal_confirmation_bonus(evaluation, true)
            } else if formal_failed {
                warn!(
                    "Formal verification FAILED for {} — applying penalty (risk realized)",
                    entity_id
                );
                apply_formal_verification_failure_penalty(evaluation)
            } else {
                compute_gate_effects(evaluation)
            };

            // Apply effects
            entity.resource_stock =
                (entity.resource_stock * effects.resource_multiplier).max(1.0);
            entity.stability =
                (entity.stability * effects.evolution_stability).clamp(0.3, 1.8);
            entity.cooperation =
                (entity.cooperation * effects.cooperation_bonus).clamp(0.2, 2.0);
            entity.information_accuracy = (entity.information_accuracy
                * effects.information_accuracy)
                .clamp(0.4, 1.6);
            entity.geometry_stability = (entity.geometry_stability
                * effects.geometry_structural_bonus)
                .clamp(0.3, 1.9);
            entity.morale = (entity.morale * effects.morale_bonus).clamp(0.3, 2.0);

            // Extra evolution progress only on successful formal confirmation
            if used_formal && all_gates_strong {
                let avg = (evaluation.score.mercy
                    + evaluation.score.harmony
                    + evaluation.score.geometry_resonance)
                    / 3.0;
                let extra = ((avg - 0.85).max(0.0) * 2.8) as u32;
                entity.evolution_stage = entity.evolution_stage.saturating_add(1 + extra);
            }

            // Emit events
            let gate_events = emit_and_collect_gate_events(evaluation, &entity_id);
            for ev in &gate_events {
                match ev {
                    GateEvent::MajorSynergy { message, .. } => info!("[GATE EVENT] {}", message),
                    GateEvent::LowGatePenalty { message, .. } => {
                        warn!("[GATE EVENT] {}", message)
                    }
                    GateEvent::ExceptionalGateHealth { message, .. } => {
                        info!("[GATE EVENT] {}", message)
                    }
                    GateEvent::FormalVerificationFailed { message, .. } => {
                        warn!("[GATE EVENT] {}", message)
                    }
                    GateEvent::FormalVerificationSuccess { message, .. } => {
                        info!("[GATE EVENT] {}", message)
                    }
                }
            }

            debug!(
                "Tick {} | formal_used={} | failed={} | overall={:.2} | resource_x={:.2}",
                entity_id,
                used_formal,
                formal_failed,
                evaluation.score.overall,
                effects.resource_multiplier
            );
        }
    }
}

// =============================================================================
// Convenience wrappers
// =============================================================================

pub fn apply_resource_generation(evaluation: &MercyGeometryEvaluation, base_amount: f32) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_amount * effects.resource_multiplier).max(0.1)
}

pub fn apply_evolution_stability(
    evaluation: &MercyGeometryEvaluation,
    base_stability: f32,
) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_stability * effects.evolution_stability).clamp(0.3, 1.8)
}

pub fn apply_cooperation_bonus(
    evaluation: &MercyGeometryEvaluation,
    base_cooperation: f32,
) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_cooperation * effects.cooperation_bonus
}

pub fn apply_information_accuracy(
    evaluation: &MercyGeometryEvaluation,
    base_accuracy: f32,
) -> f32 {
    let effects = compute_gate_effects(evaluation);
    (base_accuracy * effects.information_accuracy).clamp(0.4, 1.6)
}

pub fn apply_geometry_structural_bonus(
    evaluation: &MercyGeometryEvaluation,
    base_stability: f32,
) -> f32 {
    let effects = compute_gate_effects(evaluation);
    base_stability * effects.geometry_structural_bonus
}

/// Structured debug info for telemetry dashboards.
pub fn get_gate_debug_info(
    evaluation: &MercyGeometryEvaluation,
    entity_id: &str,
    formal_used: bool,
    formal_success: bool,
) -> GateDebugInfo {
    let s = &evaluation.score;
    let overall = s.overall as f32;

    let dominant = if s.mercy as f32 > s.love as f32 && s.mercy as f32 > s.truth as f32 {
        "Mercy"
    } else if s.love as f32 > s.harmony as f32 {
        "Love"
    } else if s.geometry_resonance as f32 > 0.8 {
        "Geometry"
    } else {
        "Harmony"
    };

    GateDebugInfo {
        entity_id: entity_id.to_string(),
        overall_gate_health: overall,
        dominant_gate: dominant.to_string(),
        synergy_level: if s.love as f32 > 0.8 && s.harmony as f32 > 0.8 {
            1.18
        } else {
            1.0
        },
        diminishing_applied: overall > 1.15,
        formal_path_used: formal_used,
        formal_success,
        message: format!(
            "Gate health {:.1}% | dominant: {} | formal: {}",
            overall * 100.0,
            dominant,
            formal_used
        ),
    }
}
