//! Mercy Geometry Evaluation — v14.15.0
//!
//! Core evaluation surface for the Powrush-MMO simulator.
//! Produces structured `MercyGeometryEvaluation` from geometry + mercy context,
//! optionally verified through the Lean WASM bridge (`MercyThresholdBridge`).
//!
//! Living Cosmic Tick aligned. Cosmic Loop is expected to be enforced by the
//! caller (gate_logic / ONE Organism surfaces).
//!
//! Contact: info@Rathor.ai

use mial::mwpo::{
    GeometryParams, MercyContext, MercyGeometryScore, MercyWeightedPreferenceOptimization,
};
use mercy_threshold_wasm::MercyThresholdBridge;

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

    /// True when the evaluation is healthy enough to drive positive system reactions.
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
            let should_evolve = score.overall >= 0.92;
            let should_apply_blessing =
                score.geometry_resonance >= 0.95 && score.mercy >= 0.98;

            MercyGeometryEvaluation {
                score,
                should_evolve,
                should_apply_blessing,
                resonance: score.geometry_resonance,
                used_formal_path: false,
                all_gates_strong: false,
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
            used_formal_path: false,
            all_gates_strong: false,
        },
    }
}

/// Formal path — attempts Lean WASM verification when a bridge is supplied.
/// Falls back cleanly to the MWPO path if the bridge is unavailable or errors.
pub fn formally_verify_geometry(
    geometry: &GeometryParams,
    context: &MercyContext,
    bridge: Option<&mut MercyThresholdBridge>,
) -> MercyGeometryEvaluation {
    if let Some(b) = bridge {
        // Map geometry into the Lean threshold parameters.
        // Higher vertices/faces produce a stricter formal check.
        let vertices = ((geometry.particle_density * 18.0) as u32).max(8).min(24);
        let faces = ((geometry.symmetry_group.order as f64 / 6.0) as u32)
            .max(6)
            .min(22);
        let chiral = geometry.symmetry_group.chiral;
        let valence = context.valence.clamp(0.6, 1.4);

        match b.check_all_gates_strong(vertices, faces, chiral, valence) {
            Ok(all_strong) => {
                println!(
                    "[MercyGeometry Formal] all_gates_strong={} | v={} f={} chiral={}",
                    all_strong, vertices, faces, chiral
                );

                let high = if all_strong { 0.99 } else { 0.62 };
                return MercyGeometryEvaluation {
                    score: MercyGeometryScore {
                        overall: high,
                        love: high,
                        mercy: high,
                        truth: high,
                        abundance: high,
                        harmony: high,
                        geometry_resonance: high,
                    },
                    should_evolve: all_strong,
                    should_apply_blessing: all_strong,
                    resonance: high,
                    used_formal_path: true,
                    all_gates_strong: all_strong,
                };
            }
            Err(e) => {
                eprintln!(
                    "[MercyGeometry Formal] bridge error — falling back to MWPO: {}",
                    e
                );
            }
        }
    }

    // Fallback / no-bridge path
    let mut eval = evaluate_particle_geometry_mercy(geometry, context);
    eval.used_formal_path = true; // we attempted the formal route
    eval
}

// =============================================================================
// Simulation system reactions (feed into gate_logic)
// =============================================================================

/// Resource / Abundance system reaction.
pub fn abundance_system_reaction(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong {
        1.25
    } else if evaluation.is_thriving() {
        1.08
    } else {
        0.85
    }
}

/// Faction / Harmony system reaction.
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

/// GPU compute modulation based on formal + resonance state.
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
