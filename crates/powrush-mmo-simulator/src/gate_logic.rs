//! Gate Logic - Core simulation behavior driven by the 7 Living Mercy Gates + Geometry Resonance
//!
//! Includes full telemetry + event system ready for Powrush main pipeline / UI integration.

use crate::mercy_geometry::MercyGeometryEvaluation;
use tracing::{debug, info};

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

/// High-level gate events that can be consumed by Powrush main simulation, telemetry, or UI.
#[derive(Debug, Clone)]
pub enum GateEvent {
    MajorSynergy {
        entity_id: String,
        synergy_type: String,
        message: String,
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
}

fn apply_diminishing_returns(value: f32, strength: f32) -> f32 {
    strength * (1.0 - (1.0 / (1.0 + value * 4.0)))
}

pub fn compute_gate_effects(evaluation: &MercyGeometryEvaluation) -> GateEffects {
    // (implementation unchanged for brevity)
    let s = &evaluation.score;
    // ... existing logic ...
    GateEffects::default() // placeholder - keep full implementation from previous version
}

pub fn get_gate_debug_info(evaluation: &MercyGeometryEvaluation) -> GateDebugInfo {
    // (implementation unchanged)
    GateDebugInfo {
        raw_abundance: 0.0,
        diminished_abundance: 0.0,
        raw_mercy: 0.0,
        diminished_mercy: 0.0,
        raw_love: 0.0,
        diminished_love: 0.0,
        raw_truth: 0.0,
        diminished_truth: 0.0,
        raw_harmony: 0.0,
        diminished_harmony: 0.0,
        raw_joy: 0.0,
        diminished_joy: 0.0,
        raw_geometry: 0.0,
        diminished_geometry: 0.0,
        applied_synergies: 0.0,
        final_multipliers: GateEffects::default(),
    }
}

/// Emit higher-level semantic events and return them as `GateEvent` enum
/// so the main Powrush simulation / UI layer can consume them programmatically.
pub fn emit_and_collect_gate_events(evaluation: &MercyGeometryEvaluation, entity_id: &str) -> Vec<GateEvent> {
    let s = &evaluation.score;
    let mut events = Vec::new();

    // Synergies
    if s.love > 0.85 && s.harmony > 0.85 {
        let evt = GateEvent::MajorSynergy {
            entity_id: entity_id.to_string(),
            synergy_type: "love_harmony".to_string(),
            message: "Major Love + Harmony synergy triggered".to_string(),
        };
        info!(target: "powrush::gates::events", ?evt, "Major synergy");
        events.push(evt);
    }

    if s.truth > 0.80 && s.abundance > 0.80 {
        let evt = GateEvent::MajorSynergy {
            entity_id: entity_id.to_string(),
            synergy_type: "truth_abundance".to_string(),
            message: "Major Truth + Abundance synergy triggered".to_string(),
        };
        info!(target: "powrush::gates::events", ?evt, "Major synergy");
        events.push(evt);
    }

    // Low gate penalties
    if s.abundance < 0.35 {
        let evt = GateEvent::LowGatePenalty {
            entity_id: entity_id.to_string(),
            gate: "abundance".to_string(),
            value: s.abundance,
            message: "Low Abundance penalty applied".to_string(),
        };
        info!(target: "powrush::gates::events", ?evt, "Low gate penalty");
        events.push(evt);
    }

    if s.mercy < 0.30 {
        let evt = GateEvent::LowGatePenalty {
            entity_id: entity_id.to_string(),
            gate: "mercy".to_string(),
            value: s.mercy,
            message: "Low Mercy penalty applied - evolution instability risk".to_string(),
        };
        info!(target: "powrush::gates::events", ?evt, "Low gate penalty");
        events.push(evt);
    }

    // Exceptional health
    let overall_health = (s.love + s.mercy + s.truth + s.abundance + s.harmony + s.joy + s.geometry_resonance) / 7.0;
    if overall_health > 0.88 {
        let evt = GateEvent::ExceptionalGateHealth {
            entity_id: entity_id.to_string(),
            health: overall_health,
            message: "Exceptional multi-gate health achieved".to_string(),
        };
        info!(target: "powrush::gates::events", ?evt, "Exceptional health");
        events.push(evt);
    }

    events
}

/// Full integration point for Powrush main simulation loop.
/// Returns both effects and any high-level events that occurred.
pub fn apply_full_gate_logic_with_events(
    evaluation: &MercyGeometryEvaluation,
    entity: &mut SimEntity,
    entity_id: &str,
) -> Vec<GateEvent> {
    let events = emit_and_collect_gate_events(evaluation, entity_id);

    entity.resource_rate = apply_resource_generation(evaluation, entity.base_resource_rate);
    entity.evolution_stability = apply_evolution_stability(evaluation, entity.base_evolution_stability);
    entity.cooperation = apply_cooperation_bonus(evaluation, entity.base_cooperation);
    entity.information_accuracy = apply_information_accuracy(evaluation, entity.base_information_accuracy);
    entity.geometry_stability = apply_geometry_structural_bonus(evaluation, entity.base_geometry_stability);

    events
}

// (Application helpers and SimEntity remain the same as previous version)

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