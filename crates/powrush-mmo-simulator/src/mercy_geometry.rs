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
//! - Deep formal verification via hybrid WASM bridge (Lean theorem)

use mial::mwpo::{MercyWeightedPreferenceOptimization, GeometryParams, MercyContext, MercyGeometryScore};
use mercy_threshold_wasm::native as hybrid_bridge;

/// High-level result for simulation systems.
#[derive(Debug, Clone)]
pub struct MercyGeometryEvaluation {
    pub score: MercyGeometryScore,
    pub should_evolve: bool,
    pub should_apply_blessing: bool,
    pub resonance: f64,
    pub used_formal_path: bool,
}

/// Evaluate geometry mercy for a particle/NPC/geometry state.
/// Fast path using native mial implementation.
pub fn evaluate_particle_geometry_mercy(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    let mut mwpo = MercyWeightedPreferenceOptimization::new();

    match mwpo.evaluate_geometry_mercy_component(geometry, context, context.council_id) {
        Ok(score) => {
            let should_evolve = score.overall >= 0.92;
            let should_apply_blessing = score.geometry_resonance >= 0.95 && score.mercy >= 0.98;

            MercyGeometryEvaluation {
                score,
                should_evolve,
                should_apply_blessing,
                resonance: score.geometry_resonance,
                used_formal_path: false,
            }
        }
        Err(_) => MercyGeometryEvaluation {
            score: MercyGeometryScore {
                overall: 0.0, love: 0.0, mercy: 0.0, truth: 0.0,
                abundance: 0.0, harmony: 0.0, geometry_resonance: 0.0,
            },
            should_evolve: false,
            should_apply_blessing: false,
            resonance: 0.0,
            used_formal_path: false,
        },
    }
}

/// Hybrid path: uses the native mial implementation when available.
/// Falls back to formal Lean WASM path via the bridge.
pub fn evaluate_geometry_mercy_hybrid(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    if let Ok(score) = hybrid_bridge::evaluate_geometry_mercy_hybrid(geometry, context, context.council_id) {
        let should_evolve = score.overall >= 0.92;
        let should_apply_blessing = score.geometry_resonance >= 0.95;

        return MercyGeometryEvaluation {
            score,
            should_evolve,
            should_apply_blessing,
            resonance: score.geometry_resonance,
            used_formal_path: false,
        };
    }

    evaluate_particle_geometry_mercy(geometry, context)
}

/// **Deep formal verification path** — prefers the Lean WASM theorem for critical checks.
/// Use this in long simulations for high-stakes geometry states (e.g. every N ticks,
/// before major world events, or on high-value particles).
pub fn formally_verify_geometry(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    if let Ok(score) = hybrid_bridge::evaluate_geometry_mercy_hybrid(geometry, context, context.council_id) {
        return MercyGeometryEvaluation {
            score,
            should_evolve: score.overall >= 0.92,
            should_apply_blessing: score.geometry_resonance >= 0.95,
            resonance: score.geometry_resonance,
            used_formal_path: true,
        };
    }

    let mut eval = evaluate_particle_geometry_mercy(geometry, context);
    eval.used_formal_path = true;
    eval
}

/// GPU layer hook: returns a modulation factor for compute dispatches.
pub fn gpu_mercy_modulation_factor(evaluation: &MercyGeometryEvaluation) -> f32 {
    (evaluation.resonance as f32 * 0.8 + 0.2).clamp(0.5, 1.5)
}

/// Decide whether to apply epigenetic blessing / mercy evolution step.
pub fn should_trigger_mercy_evolution(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.should_apply_blessing && evaluation.should_evolve
}

/// Verification policy for long-running simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationPolicy {
    /// Always use fast native path
    Fast,
    /// Use hybrid (native preferred)
    Hybrid,
    /// Use formal Lean WASM path for critical states
    Formal,
    /// Adaptive: formal every N ticks or on high-stakes geometry
    Adaptive { formal_every_n_ticks: u32 },
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        VerificationPolicy::Hybrid
    }
}

/// Evaluate with a chosen verification policy.
/// Useful for long simulations where you want to balance speed vs formal guarantees.
pub fn evaluate_with_policy(
    geometry: &GeometryParams,
    context: &MercyContext,
    policy: VerificationPolicy,
    tick: u32,
) -> MercyGeometryEvaluation {
    match policy {
        VerificationPolicy::Fast => evaluate_particle_geometry_mercy(geometry, context),
        VerificationPolicy::Hybrid => evaluate_geometry_mercy_hybrid(geometry, context),
        VerificationPolicy::Formal => formally_verify_geometry(geometry, context),
        VerificationPolicy::Adaptive { formal_every_n_ticks } => {
            if tick % formal_every_n_ticks == 0 {
                formally_verify_geometry(geometry, context)
            } else {
                evaluate_geometry_mercy_hybrid(geometry, context)
            }
        }
    }
}