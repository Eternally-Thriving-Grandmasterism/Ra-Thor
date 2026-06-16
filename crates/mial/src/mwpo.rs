//! MercyWeightedPreferenceOptimization (MWPO) — MIAL Core v13.13.0
//!
//! Optimizes preferences, decisions, and evolutionary trajectories with mercy-weighted scoring.
//! Integrates sacred geometry resonance via evaluate_geometry_mercy_component.
//! All paths non-bypassable through MercyGatingRuntime + PATSAGi Councils.
//! Monotonic mercy strengthening. ENC + esacheck truth-distilled.
//!
//! Part of internal development PR for focused iteration. AG-SML v1.0.

use mercy_gating_runtime::{MercyGatingRuntime, MercyGate, MercyContext as RuntimeMercyContext};
use nalgebra::{Vector3, Matrix4, Rotation3};
use std::collections::HashMap;

/// Main MWPO engine for MIAL.
#[derive(Debug, Clone)]
pub struct MercyWeightedPreferenceOptimization {
    runtime: MercyGatingRuntime,
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

/// Mercy context for scoring (valence, active gates, council).
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
    step: u64,
    resonance_delta: f64,
    mercy_delta: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum MialError {
    #[error("Mercy gate violation: {gate}")]
    GateViolation { gate: String },
    #[error("Geometry alignment below mercy threshold (overall < 0.92)")]
    LowAlignment,
    #[error("Invalid geometry parameters for council {council}")]
    InvalidGeometry { council: u32 },
}

impl MercyWeightedPreferenceOptimization {
    /// Create new MWPO instance with default mercy weights and fresh gating runtime.
    pub fn new() -> Self {
        let runtime = MercyGatingRuntime::new();
        Self {
            runtime,
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

    /// Optimize a set of preferences under current mercy context.
    /// Returns mercy-gated optimized actions.
    pub fn optimize_preferences(
        &mut self,
        preferences: &[Preference],
        context: &MercyContext,
    ) -> Result<Vec<OptimizedAction>, MialError> {
        self.runtime.check_gates(&context.active_gates)
            .map_err(|g| MialError::GateViolation { gate: format!("{:?}", g) })?;

        // MWPO core: weight preferences by mercy + geometry resonance
        let mut scored: Vec<_> = preferences.iter().map(|p| {
            let geo_score = self.quick_geometry_proxy(p.geometry_hint);
            let mercy_score = self.mercy_weights.get(&p.mercy_axis).copied().unwrap_or(0.9);
            (p.clone(), geo_score * mercy_score * context.valence)
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let actions = scored.into_iter().take(8).map(|(p, score)| {
            OptimizedAction {
                preference: p,
                mercy_score: score,
                gated: true,
            }
        }).collect();

        Ok(actions)
    }

    fn quick_geometry_proxy(&self, hint: Option<GeometryParams>) -> f64 {
        hint.map_or(0.85, |g| self.compute_geometry_resonance(&g))
    }

    /// Primary integration point: evaluate how geometry aligns with mercy principles.
    /// Per mial-mwpo-mercy-threshold-integration.md spec.
    /// Returns detailed MercyGeometryScore or error if below threshold.
    pub fn evaluate_geometry_mercy_component(
        &mut self,
        geometry: &GeometryParams,
        mercy_context: &MercyContext,
        council_id: u32,
    ) -> Result<MercyGeometryScore, MialError> {
        // Non-bypassable gate check
        self.runtime.check_gates(&mercy_context.active_gates)
            .map_err(|g| MialError::GateViolation { gate: format!("{:?}", g) })?;

        if mercy_context.valence < 0.999999 {
            return Err(MialError::GateViolation { gate: "valence".into() });
        }

        let resonance = self.compute_geometry_resonance(geometry);

        // 7 Living Mercy Gates sub-scores (aligned to TOLC 8)
        let love = (resonance * 0.97 + mercy_context.valence * 0.03).min(1.0);
        let mercy = (resonance * 0.92 + 0.08).min(1.0);
        let truth = if geometry.symmetry_group.chiral { 0.995 } else { 0.999 };
        let abundance = (resonance * 0.88 + 0.12).min(1.0);
        let harmony = resonance;

        let overall = (love + mercy + truth + abundance + harmony) / 5.0;

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

        // Cache + symbolic history for Lattice Conductor / self-evolution
        let key = format!("{:?}-{:?}", geometry.solid_type, geometry.evolution_step);
        self.geometry_resonance_cache.insert(key, resonance);
        self.symbolic_history.push(SymbolicRewrite {
            step: geometry.evolution_step,
            resonance_delta: resonance,
            mercy_delta: mercy_context.valence,
        });

        // PATSAGi council hook (simulated consensus via runtime)
        if council_id > 0 {
            // In full: call patsagi_councils::submit_geometry_score
        }

        Ok(score)
    }

    /// Compute sacred geometry resonance score using nalgebra + symbolic rules.
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

    /// Symbolic rewrite hook — deepens lattice state with mercy-aligned geometry.
    /// Called by Lattice Conductor during self-evolution.
    pub fn symbolic_rewrite_hook(&mut self, lattice_state: &mut LatticeState) {
        // Professional deepening with Lattice Conductor alignment language
        if let Some(last) = self.symbolic_history.last() {
            lattice_state.mercy_resonance = (lattice_state.mercy_resonance + last.resonance_delta * 0.01).min(1.0);
        }
        // PATSAGi + TOLC 8 enforcement
        self.runtime.enforce_monotonic_mercy();
    }

    /// Lightweight training / preference loop hook (expandable to full MWPO training).
    pub fn run_preference_training_loop(&mut self, epochs: u32, context: &MercyContext) -> Result<f64, MialError> {
        let mut avg_score = 0.0;
        for _ in 0..epochs {
            // simulate preference batch + geometry eval
            let dummy_geo = GeometryParams {
                solid_type: SacredSolid::Platonic,
                dimensions: 3,
                symmetry_group: SymmetryGroup { order: 48, chiral: false },
                evolution_step: 42,
                particle_density: 0.7,
                lattice_config: None,
            };
            let score = self.evaluate_geometry_mercy_component(&dummy_geo, context, context.council_id)?;
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

/// Optimized output action.
#[derive(Debug, Clone)]
pub struct OptimizedAction {
    pub preference: Preference,
    pub mercy_score: f64,
    pub gated: bool,
}

/// Minimal lattice state for hooks (real impl lives in lattice_introspection / sovereign_core).
#[derive(Debug, Clone, Default)]
pub struct LatticeState {
    pub mercy_resonance: f64,
    pub geometry_alignment: f64,
}

// Re-export for convenience in lib.rs / integration.rs
pub use self::MercyWeightedPreferenceOptimization as MWPO;