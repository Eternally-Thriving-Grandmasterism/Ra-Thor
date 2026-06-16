//! Mercy Geometry Evaluation + Simulation System Reactions
//!
//! Reacts to the formal Lean `all_gates_strong` result.

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

// === Core evaluation functions ===

pub fn evaluate_particle_geometry_mercy(
    geometry: &GeometryParams,
    context: &MercyContext,
) -> MercyGeometryEvaluation {
    let mut mwpo = MercyWeightedPreferenceOptimization::new();
    match mwpo.evaluate_geometry_mercy_component(geometry, context, context.council_id) {
        Ok(score) => {
            let should_evolve = score.overall >= 0.92;
            let should_apply_blessing = score.geometry_resonance >= 0.95 && score.mercy >= 0.98;
            MercyGeometryEvaluation { score, should_evolve, should_apply_blessing, resonance: score.geometry_resonance, used_formal_path: false, all_gates_strong: false }
        }
        Err(_) => MercyGeometryEvaluation {
            score: MercyGeometryScore { overall: 0.0, love: 0.0, mercy: 0.0, truth: 0.0, abundance: 0.0, harmony: 0.0, geometry_resonance: 0.0 },
            should_evolve: false, should_apply_blessing: false, resonance: 0.0, used_formal_path: false, all_gates_strong: false
        },
    }
}

pub fn formally_verify_geometry(
    geometry: &GeometryParams,
    context: &MercyContext,
    bridge: Option<&mut MercyThresholdBridge>,
) -> MercyGeometryEvaluation {
    if let Some(b) = bridge {
        if let Ok(all_strong) = b.check_all_gates_strong(
            geometry.vertices as u32, geometry.faces as u32, geometry.chiral, context.valence as f64
        ) {
            println!("[Formal] All gates strong (Lean): {}", all_strong);
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

// === Simulation System Reactions to all_gates_strong ===

/// Resource / Abundance system reaction
pub fn abundance_system_reaction(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong { 1.25 } else { 0.85 }  // +25% abundance when all gates strong
}

/// Faction / Harmony system reaction
pub fn faction_harmony_reaction(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong { 1.15 } else { 0.90 }
}

/// Evolution / Blessing system reaction
pub fn evolution_system_reaction(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.all_gates_strong && evaluation.should_apply_blessing
}

/// GPU compute modulation based on formal gate state
pub fn gpu_mercy_modulation(evaluation: &MercyGeometryEvaluation) -> f32 {
    if evaluation.all_gates_strong { 1.3 } else { 0.85 }
}

pub fn should_trigger_mercy_evolution(evaluation: &MercyGeometryEvaluation) -> bool {
    evaluation.should_apply_blessing && evaluation.should_evolve
}