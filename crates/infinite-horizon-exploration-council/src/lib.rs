//! Council record.
//! The type name and `name` string are a historical title (design intent).
//!
//! `src/uncompiled_council.rs` is not a module.
//! The bytes under its design-only header are the previous `src/lib.rs`.
//!
//! Compiled call: `RathorSovereignReasoningEngine::evaluate_proposal`, then
//! `run_foresight_simulation` when `proposal_valence` is at least the engine
//! source literal `MERCY_THRESHOLD` (0.9999999).
//! The simulation caps are source literals in that engine.
//! `mean_bond_valence` is the mean of caller bond valences that `fuse_bond` stores.
//! The philotic rejection line is the source literal 0.9999999.
//! `apply_shift` is the Möbius image `(z + shift)`.
//! `disk_distance` is the Poincaré-disk distance of the two caller points.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use hyperbolic_tiling_consciousness::{hyperbolic_distance, PoincarePoint};
use moebius_transformations::MoebiusMatrix;
use nalgebra::Complex;
use philotic_web_fusion::PhiloticWeb;
use rathor_sovereign_reasoning_engine::RathorSovereignReasoningEngine;

/// Source literal from `rathor-sovereign-reasoning-engine`. Not a measured value.
pub use rathor_sovereign_reasoning_engine::MERCY_THRESHOLD as REASONING_THRESHOLD_SOURCE_LITERAL;

/// Historical council title (design intent).
pub struct InfiniteHorizonExplorationCouncil {
    pub id: u8,
    pub name: String,
    /// Source literal `0.9999999`. Not a measured valence.
    valence_threshold: f64,
}

impl InfiniteHorizonExplorationCouncil {
    pub fn new() -> Self {
        Self {
            id: 16,
            name: "Infinite Horizon Exploration Council".to_string(),
            valence_threshold: 0.9999999,
        }
    }

    /// Returns the source literal `0.9999999`. Not a measured valence.
    pub fn valence_threshold(&self) -> f64 {
        self.valence_threshold
    }

    /// `valence >= valence_threshold`.
    /// `valence_threshold` is the source literal 0.9999999, not a measured valence.
    pub fn tolc8_mercy_check(&self, valence: f64) -> bool {
        valence >= self.valence_threshold
    }

    /// Below `REASONING_THRESHOLD_SOURCE_LITERAL`, returns the engine Err text.
    /// Otherwise returns `run_foresight_simulation(years)`.
    /// Both caps inside that simulation are source literals.
    /// The engine's second argument stays false. This function does not return the engine Ok text.
    pub fn reasoning_from_inputs(proposal_valence: f64, years: u64) -> Result<f64, String> {
        let mut engine = RathorSovereignReasoningEngine::new();
        engine.evaluate_proposal(proposal_valence, false)?;
        Ok(engine.run_foresight_simulation(years))
    }
}

impl Default for InfiniteHorizonExplorationCouncil {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean valence of the supplied bonds.
/// An empty slice returns `Err("no bonds supplied")`.
/// A valence below the philotic source literal 0.9999999 returns that crate's Err text.
pub fn mean_bond_valence(bonds: &[(&str, &str, f64)]) -> Result<f64, String> {
    if bonds.is_empty() {
        return Err("no bonds supplied".to_string());
    }
    let mut web = PhiloticWeb::new();
    for (from, to, valence) in bonds {
        web.fuse_bond(from, to, 1.0, *valence)?;
    }
    Ok(web.web_valence())
}

/// Möbius image of `z` under `a = 1`, `b = shift`, `c = 0`, `d = 1`.
pub fn apply_shift(z_re: f64, z_im: f64, shift: f64) -> (f64, f64) {
    let map = MoebiusMatrix::new(
        Complex::new(1.0, 0.0),
        Complex::new(shift, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    );
    let image = map.apply(Complex::new(z_re, z_im));
    (image.re, image.im)
}

/// Poincaré-disk distance. Both points must satisfy `x^2 + y^2 < 1`.
pub fn disk_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let left = PoincarePoint::new(ax, ay);
    let right = PoincarePoint::new(bx, by);
    hyperbolic_distance(&left, &right)
}
