//! `MercyWeightedPreferenceOptimization` scores geometry with the formulas in this file.
//! The numbers in `compute_geometry_resonance` and `default_mercy_weights` are source literals.
//! They are not measured results.
//! `evaluate_geometry_mercy_component` rejects `valence` below the source literal `0.999999`.
//! It does not read `MercyContext::active_gates`.
//! This type does not call `MercyGatingRuntime`.
//! `optimize_preferences` stores `gated` as false. No compiled check in this file sets it true.
//! Contact: info@Rathor.ai

use nalgebra::Matrix4;
use std::collections::HashMap;

/// Preference scorer. Resonance figures are source literals from `compute_geometry_resonance`.
#[derive(Debug, Clone)]
pub struct MercyWeightedPreferenceOptimization {
    mercy_weights: HashMap<String, f64>,
    geometry_resonance_cache: HashMap<String, f64>,
    symbolic_history: Vec<SymbolicRewrite>,
}

/// Geometry parameters for mercy-aligned evaluation (sacred geometry, particle evolution, lattices).
#[derive(Debug, Clone)]
pub struct GeometryParams {
    pub solid_type: SacredSolid,
    pub dimensions: u8,
    pub symmetry_group: SymmetryGroup,
    pub evolution_step: u64,
    pub particle_density: f64,
    pub lattice_config: Option<Matrix4<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SacredSolid {
    Platonic,
    Archimedean,
    Johnson,
    Catalan,
    Disdyakis,
    KeplerPoinsot,
    UniformStar,
    Hyperbolic,
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetryGroup {
    pub order: u32,
    pub chiral: bool,
}

/// Label stored on `MercyContext::active_gates`.
/// `evaluate_geometry_mercy_component` does not read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGate {
    Truth,
    Love,
    Mercy,
    Abundance,
    Harmony,
    CosmicHarmony,
    BoundlessMercy,
}

/// Stored valence, gate labels, and a council id.
/// `active_gates` is not read by the compiled functions in this file.
#[derive(Debug, Clone)]
pub struct MercyContext {
    pub active_gates: Vec<MercyGate>,
    pub valence: f64,
    pub council_id: u32,
}

/// Result of geometry mercy evaluation.
#[derive(Debug, Clone)]
pub struct MercyGeometryScore {
    pub overall: f64,
    pub love: f64,
    pub mercy: f64,
    pub truth: f64,
    pub abundance: f64,
    pub harmony: f64,
    pub geometry_resonance: f64,
}

#[derive(Debug, Clone)]
struct SymbolicRewrite {
    resonance_delta: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum MialError {
    #[error("Mercy gate violation: {gate}")]
    GateViolation { gate: String },
    #[error("overall is below the source literal 0.92 (not a measured value)")]
    LowAlignment,
    #[error("Invalid geometry parameters for council {council}")]
    InvalidGeometry { council: u32 },
}

impl Default for MercyWeightedPreferenceOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl MercyWeightedPreferenceOptimization {
    /// Weights are the source literals in `default_mercy_weights`.
    pub fn new() -> Self {
        Self {
            mercy_weights: Self::default_mercy_weights(),
            geometry_resonance_cache: HashMap::new(),
            symbolic_history: Vec::new(),
        }
    }

    fn default_mercy_weights() -> HashMap<String, f64> {
        let mut w = HashMap::new();
        w.insert("radical_love".to_string(), 1.0);
        w.insert("boundless_mercy".to_string(), 1.05);
        w.insert("truth".to_string(), 0.98);
        w.insert("abundance".to_string(), 0.95);
        w.insert("cosmic_harmony".to_string(), 1.02);
        w
    }

    /// Scores each preference as `geometry_proxy * weight * context.valence`.
    /// A missing weight uses the source literal `0.9`.
    /// `active_gates` is not read.
    /// `gated` is false. This function does not run a check.
    pub fn optimize_preferences(
        &mut self,
        preferences: &[Preference],
        context: &MercyContext,
    ) -> Result<Vec<OptimizedAction>, MialError> {
        let mut scored: Vec<_> = preferences
            .iter()
            .map(|p| {
                let geo_score = self.quick_geometry_proxy(p.geometry_hint.clone());
                let mercy_score = self
                    .mercy_weights
                    .get(&p.mercy_axis)
                    .copied()
                    .unwrap_or(0.9);
                (p.clone(), geo_score * mercy_score * context.valence)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let actions = scored
            .into_iter()
            .take(8)
            .map(|(p, score)| {
                OptimizedAction {
                    preference: p,
                    mercy_score: score,
                    // No compiled check runs here. Fail closed.
                    gated: false,
                }
            })
            .collect();

        Ok(actions)
    }

    fn quick_geometry_proxy(&self, hint: Option<GeometryParams>) -> f64 {
        // 0.85 is a source literal used when `hint` is None. It is not a measured value.
        hint.map_or(0.85, |g| self.compute_geometry_resonance(&g))
    }

    /// Scores one geometry.
    /// Returns `GateViolation { gate: "valence" }` when `valence` is below the source literal `0.999999`.
    /// Returns `LowAlignment` when `overall` is below the source literal `0.92`.
    /// `truth` is the source literal `0.995` when `chiral` is true, and `0.999` when it is false.
    /// `active_gates` is not read. `council_id` is stored in the cache key only.
    pub fn evaluate_geometry_mercy_component(
        &mut self,
        geometry: &GeometryParams,
        mercy_context: &MercyContext,
        council_id: u32,
    ) -> Result<MercyGeometryScore, MialError> {
        // 0.999999 is a source literal, not a measured value.
        if mercy_context.valence < 0.999999 {
            return Err(MialError::GateViolation {
                gate: "valence".into(),
            });
        }

        let resonance = self.compute_geometry_resonance(geometry);

        // Coefficients 0.97, 0.03, 0.92, 0.08, 0.995, 0.999, 0.88, and 0.12 are source literals.
        let love = (resonance * 0.97 + mercy_context.valence * 0.03).min(1.0);
        let mercy = (resonance * 0.92 + 0.08).min(1.0);
        let truth = if geometry.symmetry_group.chiral {
            0.995
        } else {
            0.999
        };
        let abundance = (resonance * 0.88 + 0.12).min(1.0);
        let harmony = resonance;

        let overall = (love + mercy + truth + abundance + harmony) / 5.0;

        // 0.92 is a source literal, not a measured value.
        if overall < 0.92 {
            return Err(MialError::LowAlignment);
        }

        let score = MercyGeometryScore {
            overall,
            love,
            mercy,
            truth,
            abundance,
            harmony,
            geometry_resonance: resonance,
        };

        let key = format!(
            "{:?}-{}-{}",
            geometry.solid_type, geometry.evolution_step, council_id
        );
        self.geometry_resonance_cache.insert(key, resonance);
        self.symbolic_history.push(SymbolicRewrite {
            resonance_delta: resonance,
        });

        Ok(score)
    }

    /// Sum of the solid's source literal, a symmetry term, a density term, and an evolution term.
    /// The solid arms and the caps `0.08`, `0.15`, `0.05`, and `1.0` are source literals.
    fn compute_geometry_resonance(&self, geometry: &GeometryParams) -> f64 {
        let base = match geometry.solid_type {
            SacredSolid::Platonic => 0.96,
            SacredSolid::Archimedean => 0.94,
            SacredSolid::Johnson => 0.91,
            SacredSolid::Catalan => 0.93,
            SacredSolid::Disdyakis => 0.97,
            SacredSolid::KeplerPoinsot => 0.89,
            SacredSolid::UniformStar => 0.90,
            SacredSolid::Hyperbolic => 0.87,
        };

        let symmetry_bonus = (geometry.symmetry_group.order as f64 / 120.0).min(0.08);
        let density_factor = (geometry.particle_density * 0.12).min(0.15);
        let evolution_factor = ((geometry.evolution_step % 100) as f64 / 1000.0).min(0.05);

        (base + symmetry_bonus + density_factor + evolution_factor).min(1.0)
    }

    /// Adds `last.resonance_delta * 0.01` to `mercy_resonance`, then applies `.min(1.0)`.
    /// `0.01` and `1.0` are source literals. No gate runs in this function.
    /// An empty history leaves `lattice_state` unchanged.
    pub fn symbolic_rewrite_hook(&mut self, lattice_state: &mut LatticeState) {
        if let Some(last) = self.symbolic_history.last() {
            lattice_state.mercy_resonance =
                (lattice_state.mercy_resonance + last.resonance_delta * 0.01).min(1.0);
        }
    }

    /// Averages `epochs` calls to `evaluate_geometry_mercy_component` on one fixed geometry.
    /// `epochs == 0` returns the source literal `0.0` (there is no average).
    pub fn run_preference_training_loop(
        &mut self,
        epochs: u32,
        context: &MercyContext,
    ) -> Result<f64, MialError> {
        if epochs == 0 {
            return Ok(0.0);
        }
        let mut avg_score = 0.0;
        for _ in 0..epochs {
            let dummy_geo = GeometryParams {
                solid_type: SacredSolid::Platonic,
                dimensions: 3,
                symmetry_group: SymmetryGroup {
                    order: 48,
                    chiral: false,
                },
                evolution_step: 42,
                particle_density: 0.7,
                lattice_config: None,
            };
            let score =
                self.evaluate_geometry_mercy_component(&dummy_geo, context, context.council_id)?;
            avg_score += score.overall;
        }
        Ok(avg_score / epochs as f64)
    }
}

/// Simple preference input (expand as needed).
#[derive(Debug, Clone)]
pub struct Preference {
    pub id: u64,
    pub mercy_axis: String,
    pub weight: f64,
    pub geometry_hint: Option<GeometryParams>,
}

/// Scored preference.
/// `gated` is false unless a compiled check ran and passed.
/// `optimize_preferences` does not run a check, so it stores false.
#[derive(Debug, Clone)]
pub struct OptimizedAction {
    pub preference: Preference,
    pub mercy_score: f64,
    pub gated: bool,
}

/// Two stored f64 fields. `symbolic_rewrite_hook` writes `mercy_resonance` only.
#[derive(Debug, Clone, Default)]
pub struct LatticeState {
    pub mercy_resonance: f64,
    pub geometry_alignment: f64,
}

pub use self::MercyWeightedPreferenceOptimization as MWPO;

#[cfg(test)]
mod tests {
    use super::*;

    fn platonic_zero() -> GeometryParams {
        GeometryParams {
            solid_type: SacredSolid::Platonic,
            dimensions: 3,
            symmetry_group: SymmetryGroup {
                order: 0,
                chiral: false,
            },
            evolution_step: 0,
            particle_density: 0.0,
            lattice_config: None,
        }
    }

    fn context(valence: f64) -> MercyContext {
        MercyContext {
            active_gates: vec![MercyGate::Truth],
            valence,
            council_id: 13,
        }
    }

    #[test]
    fn platonic_zero_resonance_is_the_source_literal_0_96() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let score = mwpo
            .evaluate_geometry_mercy_component(&platonic_zero(), &context(0.999999), 13)
            .expect("overall is above 0.92");
        assert_eq!(score.geometry_resonance, 0.96);
        assert_eq!(score.harmony, 0.96);
        assert_eq!(score.truth, 0.999);
        assert_eq!(score.love, (0.96_f64 * 0.97 + 0.999999 * 0.03).min(1.0));
        assert_eq!(score.mercy, (0.96_f64 * 0.92 + 0.08).min(1.0));
        assert_eq!(score.abundance, (0.96_f64 * 0.88 + 0.12).min(1.0));
        let overall =
            (score.love + score.mercy + score.truth + score.abundance + score.harmony) / 5.0;
        assert_eq!(score.overall, overall);
        assert!(score.overall >= 0.92);
        assert_eq!(
            mwpo.geometry_resonance_cache.get("Platonic-0-13"),
            Some(&0.96)
        );
    }

    #[test]
    fn chiral_truth_is_the_source_literal_0_995() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let mut geometry = platonic_zero();
        geometry.symmetry_group.chiral = true;
        let score = mwpo
            .evaluate_geometry_mercy_component(&geometry, &context(0.999999), 1)
            .expect("overall is above 0.92");
        assert_eq!(score.truth, 0.995);
    }

    #[test]
    fn valence_below_the_source_literal_is_rejected() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let err = mwpo
            .evaluate_geometry_mercy_component(&platonic_zero(), &context(0.999998), 1)
            .expect_err("0.999998 is below 0.999999");
        match err {
            MialError::GateViolation { gate } => assert_eq!(gate, "valence"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn hyperbolic_zero_is_below_the_source_literal_0_92() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let mut geometry = platonic_zero();
        geometry.solid_type = SacredSolid::Hyperbolic;
        let err = mwpo
            .evaluate_geometry_mercy_component(&geometry, &context(0.999999), 1)
            .expect_err("hyperbolic base 0.87 stays under 0.92");
        assert!(matches!(err, MialError::LowAlignment));
    }

    #[test]
    fn missing_geometry_hint_uses_the_source_literal_0_85() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let actions = mwpo
            .optimize_preferences(
                &[Preference {
                    id: 7,
                    mercy_axis: "truth".into(),
                    weight: 0.5,
                    geometry_hint: None,
                }],
                &context(1.0),
            )
            .expect("no gate runs");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].preference.id, 7);
        assert_eq!(actions[0].mercy_score, 0.85 * 0.98 * 1.0);
        assert!(!actions[0].gated);
        assert_eq!(mwpo.mercy_weights.get("truth"), Some(&0.98));
        assert_eq!(mwpo.mercy_weights.get("radical_love"), Some(&1.0));
    }

    #[test]
    fn stored_flag_is_false_when_no_check_ran() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let actions = mwpo
            .optimize_preferences(
                &[
                    Preference {
                        id: 1,
                        mercy_axis: "truth".into(),
                        weight: 1.0,
                        geometry_hint: None,
                    },
                    Preference {
                        id: 2,
                        mercy_axis: "missing-axis".into(),
                        weight: 1.0,
                        geometry_hint: Some(platonic_zero()),
                    },
                ],
                &context(1.0),
            )
            .expect("no check runs");
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().all(|action| !action.gated));
    }

    #[test]
    fn empty_history_leaves_lattice_resonance_at_0_5() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let mut state = LatticeState {
            mercy_resonance: 0.5,
            geometry_alignment: 0.25,
        };
        mwpo.symbolic_rewrite_hook(&mut state);
        assert_eq!(state.mercy_resonance, 0.5);
        assert_eq!(state.geometry_alignment, 0.25);
    }

    #[test]
    fn zero_epochs_returns_the_source_literal_0() {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        let avg = mwpo
            .run_preference_training_loop(0, &context(0.999999))
            .expect("zero epochs");
        assert_eq!(avg, 0.0);
    }
}
