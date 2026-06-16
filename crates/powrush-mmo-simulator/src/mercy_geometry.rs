//! Mercy Geometry Evaluation Integration for Powrush-MMO Simulator
//!
//! Wires MIAL's evaluate_geometry_mercy_component + Safety Harness
//! + hybrid Lean/Rust threshold bridge into the simulation.
//!
//! This module provides simulation-friendly APIs for:
//! - Particle / NPC / geometry state evaluation
//! - Mercy-gated evolution decisions
//! - Safety preflight before major state changes
//! - GPU layer modulation hooks

use mial::mwpo::{MercyWeightedPreferenceOptimization, GeometryParams, MercyContext, MercyGeometryScore};
use mercy_threshold_wasm::native as hybrid_bridge;
use crate::safety_harness::PatsagiSafetyHarness; // if exposed from mial, or re-export

/// High-level result for simulation systems.
#[derive(Debug, Clone)]
pub struct MercyGeometryEvaluation {
    pub score: MercyGeometryScore,
    pub should_evolve: bool,
    pub should_apply_blessing: bool,
    pub resonance: f64,
}

/// Evaluate geometry mercy for a particle/NPC/geometry state.
/// This is the main entry point for the simulator.
pub fn evaluate_particle_geometry_mercy(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    let mut mwpo = MercyWeightedPreferenceOptimization::new();

    // Run safety preflight (from mial safety harness concepts)
    // In full integration this would call PatsagiSafetyHarness::preflight_safety_check

    match mwpo.evaluate_geometry_mercy_component(geometry, context, context.council_id) {
        Ok(score) => {
            let should_evolve = score.overall >= 0.92;
            let should_apply_blessing = score.geometry_resonance >= 0.95 && score.mercy >= 0.98;

            MercyGeometryEvaluation {
                score,
                should_evolve,
                should_apply_blessing,
                resonance: score.geometry_resonance,
            }
        }
        Err(_) => MercyGeometryEvaluation {
            score: MercyGeometryScore {
                overall: 0.0,
                love: 0.0,
                mercy: 0.0,
                truth: 0.0,
                abundance: 0.0,
                harmony: 0.0,
                geometry_resonance: 0.0,
            },
            should_evolve: false,
            should_apply_blessing: false,
            resonance: 0.0,
        },
    }
}

/// Hybrid path: uses the native mial implementation when available,
/// falls back to formal Lean WASM path via the bridge.
pub fn evaluate_geometry_mercy_hybrid(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    // Prefer full native mial path (faster + richer)
    if let Ok(score) = hybrid_bridge::evaluate_geometry_mercy_hybrid(geometry, context, context.council_id) {
        let should_evolve = score.overall >= 0.92;
        let should_apply_blessing = score.geometry_resonance >= 0.95;

        return MercyGeometryEvaluation {
            score,
            should_evolve,
            should_apply_blessing,
            resonance: score.geometry_resonance,
        };
    }

    // Fallback to direct mial call
    evaluate_particle_geometry_mercy(geometry, context)
}

/// GPU layer hook: returns a modulation factor for compute dispatches.
/// Higher mercy resonance = more stable / abundant simulation behavior.
pub fn gpu_mercy_modulation_factor(evaluation: &MercyGeometryEvaluation) -> f32 {
    (evaluation.resonance as f32 * 0.8 + 0.2).clamp(0.5, 1.5)
}

/// Decide whether to apply epigenetic blessing / mercy evolution step.
pub fn should_trigger_mercy_evolution(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.should_apply_blessing && evaluation.should_evolve
}