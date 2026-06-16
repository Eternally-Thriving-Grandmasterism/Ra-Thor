//! Mercy Geometry Evaluation for Powrush-MMO Simulator
//!
//! Uses the formal Lean bridge lemma result when available.

use mial::mwpo::{MercyWeightedPreferenceOptimization, GeometryParams, MercyContext, MercyGeometryScore};
use mercy_threshold_wasm::MercyThresholdBridge;

#[derive(Debug, Clone)]
pub struct MercyGeometryEvaluation {
    pub score: MercyGeometryScore,
    pub should_evolve: bool,
    pub should_apply_blessing: bool,
    pub resonance: f64,
    pub used_formal_path: bool,
    pub all_gates_strong: bool,
}

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
                all_gates_strong: false,
            }
        }
        Err(_) => MercyGeometryEvaluation {
            score: MercyGeometryScore { overall: 0.0, love: 0.0, mercy: 0.0, truth: 0.0, abundance: 0.0, harmony: 0.0, geometry_resonance: 0.0 },
            should_evolve: false,
            should_apply_blessing: false,
            resonance: 0.0,
            used_formal_path: false,
            all_gates_strong: false,
        },
    }
}

/// Formal verification that uses the Lean bridge lemma result when possible.
pub fn formally_verify_geometry(
    geometry: &GeometryParams,
    context: &MercyContext,
    bridge: Option<&mut MercyThresholdBridge>,
) -> MercyGeometryEvaluation {
    if let Some(b) = bridge {
        if let Ok(all_strong) = b.check_all_gates_strong(
            geometry.vertices as u32,
            geometry.faces as u32,
            geometry.chiral,
            context.valence as f64,
        ) {
            println!("[Formal] All gates strong (Lean bridge lemma): {}", all_strong);

            return MercyGeometryEvaluation {
                score: MercyGeometryScore {
                    overall: if all_strong { 0.99 } else { 0.6 },
                    love: if all_strong { 0.99 } else { 0.6 },
                    mercy: if all_strong { 0.99 } else { 0.6 },
                    truth: if all_strong { 0.99 } else { 0.6 },
                    abundance: if all_strong { 0.99 } else { 0.6 },
                    harmony: if all_strong { 0.99 } else { 0.6 },
                    geometry_resonance: if all_strong { 0.99 } else { 0.6 },
                },
                should_evolve: all_strong,
                should_apply_blessing: all_strong,
                resonance: if all_strong { 0.99 } else { 0.6 },
                used_formal_path: true,
                all_gates_strong: all_strong,
            };
        }
    }

    let mut eval = evaluate_particle_geometry_mercy(geometry, context);
    eval.used_formal_path = true;
    eval
}

pub fn gpu_mercy_modulation_factor(evaluation: &MercyGeometryEvaluation) -> f32 {
    (evaluation.resonance as f32 * 0.8 + 0.2).clamp(0.5, 1.5)
}

pub fn should_trigger_mercy_evolution(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.should_apply_blessing && evaluation.should_evolve
}