//! Mercy geometry evaluation — v14.15.0
//!
//! `evaluate_particle_geometry_mercy` returns the mial MWPO score, or the `0.0`
//! failure sentinel when MWPO returns `Err`. `0.0` is a source literal, not a
//! measured value. Reaction multipliers in this file are source literals too.
//! `formally_verify_geometry` may set `all_gates_strong` from a caller-supplied
//! wasm export. It does not replace the MWPO score with another constant.
//! This file does not read MWPO `gated` and does not call `MercyGatingRuntime`.
//!
//! Contact: info@Rathor.ai

use mercy_threshold_wasm::MercyThresholdBridge;
use mial::mwpo::{GeometryParams, MercyContext, MercyWeightedPreferenceOptimization};
use tracing::warn;

// Re-export the score type so downstream crates can use it cleanly
pub use mial::mwpo::MercyGeometryScore;

// =============================================================================
// Core evaluation type
// =============================================================================

#[derive(Debug, Clone)]
pub struct MercyGeometryEvaluation {
    pub score: MercyGeometryScore,
    pub should_evolve: bool,
    pub should_apply_blessing: bool,
    pub resonance: f64,
    pub used_formal_path: bool,
    pub all_gates_strong: bool,
}

impl MercyGeometryEvaluation {
    /// Lightweight status string for telemetry / PATSAGi observation.
    pub fn summary(&self) -> String {
        format!(
            "MercyGeometry v14.15.0 | overall={:.3} | resonance={:.3} | evolve={} | blessing={} | formal={} | all_strong={}",
            self.score.overall,
            self.resonance,
            self.should_evolve,
            self.should_apply_blessing,
            self.used_formal_path,
            self.all_gates_strong
        )
    }

    /// `0.88` and `0.85` are source literals, not measured values.
    pub fn is_thriving(&self) -> bool {
        self.score.overall >= 0.88 && self.resonance >= 0.85
    }
}

// =============================================================================
// Core evaluation paths
// =============================================================================

/// Primary non-formal evaluation path (MWPO geometry component).
pub fn evaluate_particle_geometry_mercy(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    let mut mwpo = MercyWeightedPreferenceOptimization::new();
    match mwpo.evaluate_geometry_mercy_component(geometry, context, context.council_id) {
        Ok(score) => {
            // 0.92, 0.95, and 0.98 are source literals, not measured values.
            let should_evolve = score.overall >= 0.92;
            let should_apply_blessing = score.geometry_resonance >= 0.95 && score.mercy >= 0.98;
            let resonance = score.geometry_resonance;

            MercyGeometryEvaluation {
                score,
                should_evolve,
                should_apply_blessing,
                resonance,
                used_formal_path: false,
                all_gates_strong: false,
            }
        }
        Err(_) => MercyGeometryEvaluation {
            // 0.0 is a source literal for the failed MWPO call, not a measured value.
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
            used_formal_path: false,
            all_gates_strong: false,
        },
    }
}

/// Sets `all_gates_strong` from `check_all_gates_strong` when `bridge` is `Some`.
/// Scores stay the MWPO result from `evaluate_particle_geometry_mercy`.
/// This function does not replace that score with a constant.
/// This crate does not embed a proof. No bridge means `used_formal_path` stays false.
/// `0.6` and `1.4` clamp the wasm argument only. They are source literals.
pub fn formally_verify_geometry(
    geometry: &GeometryParams,
    context: &MercyContext,
    bridge: Option<&mut MercyThresholdBridge>,
) -> MercyGeometryEvaluation {
    let mut eval = evaluate_particle_geometry_mercy(geometry, context);
    let Some(bridge) = bridge else {
        return eval;
    };

    // 18.0, 8, 24, 6.0, 6, and 22 map geometry into the wasm argument ranges.
    // They are source literals, not measured values.
    let vertices = ((geometry.particle_density * 18.0) as u32).clamp(8, 24);
    let faces = ((geometry.symmetry_group.order as f64 / 6.0) as u32).clamp(6, 22);
    let chiral = geometry.symmetry_group.chiral;
    let valence = context.valence.clamp(0.6, 1.4);

    eval.used_formal_path = true;
    match bridge.check_all_gates_strong(vertices, faces, chiral, valence) {
        Ok(all_strong) => {
            eval.all_gates_strong = all_strong;
        }
        Err(err) => {
            eval.all_gates_strong = false;
            warn!("wasm export check_all_gates_strong returned an error: {err}");
        }
    }
    eval
}

// =============================================================================
// Simulation system reactions (feed into gate_logic)
// =============================================================================

/// Resource / abundance multiplier. `1.25`, `1.08`, and `0.85` are source literals.
pub fn abundance_system_reaction(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong {
        1.25
    } else if evaluation.is_thriving() {
        1.08
    } else {
        0.85
    }
}

/// Faction / harmony multiplier. `1.15`, `1.06`, `0.88`, and `0.90` are source literals.
pub fn faction_harmony_reaction(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong {
        1.15
    } else if evaluation.score.harmony >= 0.88 {
        1.06
    } else {
        0.90
    }
}

/// Evolution / Blessing system reaction.
pub fn evolution_system_reaction(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.all_gates_strong && evaluation.should_apply_blessing
}

/// Compute modulation. `1.30`, `1.12`, `0.92`, and `0.85` are source literals.
pub fn gpu_mercy_modulation(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong {
        1.30
    } else if evaluation.resonance >= 0.92 {
        1.12
    } else {
        0.85
    }
}

/// Convenience trigger used by higher-level simulators.
pub fn should_trigger_mercy_evolution(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.should_apply_blessing && evaluation.should_evolve
}

/// Composite health score useful for telemetry dashboards.
pub fn composite_gate_health(evaluation: &MercyGeometryEvaluation) -> f32 {
    let s = &evaluation.score;
    ((s.love + s.mercy + s.truth + s.abundance + s.harmony + s.geometry_resonance) / 6.0) as f32
}
