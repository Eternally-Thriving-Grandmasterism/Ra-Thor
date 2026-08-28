//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient · valence · adaptive floor · concurrent zones · soft feedback · LatticeHealthReport · adaptive Cosmic Tick · zone observability · stress EMA recovery · health aggregates · composite score · ZoneHealthStatus · critical auto-remediate · valence histogram · soft-remediate Stressed · Net Eternal Valence Contribution (NEVC) · Tikhonov-damped projector
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

mod soft_feedback;
pub use soft_feedback::*;

mod nevc;
pub use nevc::*;

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

pub const AMBIENT_DIM: usize = 16;
pub const MERCY_DIM: usize = 8;
pub const MERCY_PURITY_FLOOR: f64 = 1e-9;
/// Gain from Gram residual ρ = ‖EᵀE − I₈‖ into the Tikhonov floor λ.
pub const TIKHONOV_RHO_GAIN: f64 = 1.0;
/// Gain from zone stress EMA into λ (kept small so calm zones stay exact).
pub const TIKHONOV_STRESS_GAIN: f64 = 1e-6;

/// Continuous floor under the Gram inverse.
/// Canonical / purified frames (ρ ≈ 0, calm stress) keep λ = 0 so P is exact.
/// Drift or stress raises λ; `purify` drives ρ → 0 and therefore resets λ.
pub fn schedule_tikhonov_lambda(rho: f64, stress_ema: f64) -> f64 {
    TIKHONOV_RHO_GAIN * rho.max(0.0) + TIKHONOV_STRESS_GAIN * stress_ema.max(0.0)
}

pub fn gram_residual(e: &MercyBasisMatrix) -> f64 {
    let gram: MercyGram = e.transpose() * e;
    (gram - MercyGram::identity()).norm()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Valence(pub f64);

impl Valence {
    pub const HIGH: Valence = Valence(0.999999);
    pub const MID: Valence = Valence(0.5);
    pub const ZERO: Valence = Valence(0.0);
    pub fn new(v: f64) -> Self { Valence(v.clamp(0.0, 1.0)) }
    pub fn value(self) -> f64 { self.0 }
    pub fn deficit(self) -> f64 { 1.0 - self.0 }
    pub fn is_high(self) -> bool { self.0 >= 0.999999 }
    pub fn purity_floor(self) -> f64 { adaptive_purity_floor(self) }
}

impl Default for Valence {
    fn default() -> Self { Valence::HIGH }
}

pub fn adaptive_purity_floor(valence: Valence) -> f64 {
    MERCY_PURITY_FLOOR * (1.0 + 99.0 * valence.deficit())
}

pub type AmbientVector = SVector<f64, AMBIENT_DIM>;
pub type AmbientMatrix = SMatrix<f64, AMBIENT_DIM, AMBIENT_DIM>;
pub type MercyBasisMatrix = SMatrix<f64, AMBIENT_DIM, MERCY_DIM>;
pub type MercyGram = SMatrix<f64, MERCY_DIM, MERCY_DIM>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGate {
    Truth, Order, Love, Compassion, Service, Abundance, Joy, CosmicHarmony,
}

impl MercyGate {
    pub const ALL: [MercyGate; MERCY_DIM] = [
        MercyGate::Truth, MercyGate::Order, MercyGate::Love, MercyGate::Compassion,
        MercyGate::Service, MercyGate::Abundance, MercyGate::Joy, MercyGate::CosmicHarmony,
    ];
    pub fn index(self) -> usize {
        match self {
            MercyGate::Truth => 0, MercyGate::Order => 1, MercyGate::Love => 2,
            MercyGate::Compassion => 3, MercyGate::Service => 4, MercyGate::Abundance => 5,
            MercyGate::Joy => 6, MercyGate::CosmicHarmony => 7,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivingMercyBasis {
    pub e: MercyBasisMatrix,
    /// Ridge λ for P_λ = E(EᵀE + λI)⁻¹Eᵀ. Zero on a purified frame.
    #[serde(default)]
    pub tikhonov_lambda: f64,
}

impl Default for LivingMercyBasis {
    fn default() -> Self { Self::canonical() }
}

impl LivingMercyBasis {
    pub fn canonical() -> Self {
        let mut e = MercyBasisMatrix::zeros();
        for i in 0..MERCY_DIM { e[(i, i)] = 1.0; }
        Self { e, tikhonov_lambda: 0.0 }
    }
    pub fn identity() -> Self { Self::canonical() }
    pub fn gram_residual(&self) -> f64 { gram_residual(&self.e) }
    pub fn reschedule_tikhonov(&mut self, stress_ema: f64) -> f64 {
        self.tikhonov_lambda = schedule_tikhonov_lambda(self.gram_residual(), stress_ema);
        self.tikhonov_lambda
    }
    /// Exact (undamped) projector: binary `try_inverse` or EEᵀ fallback.
    pub fn projector_matrix_exact(&self) -> AmbientMatrix {
        let et = self.e.transpose();
        let gram: MercyGram = et * &self.e;
        match gram.try_inverse() {
            Some(ginv) => &self.e * ginv * et,
            None => &self.e * et,
        }
    }
    /// Damped projector P_λ = E(EᵀE + λI)⁻¹Eᵀ.
    /// Symmetric and positive semidefinite for λ ≥ 0; idempotent only at λ = 0
    /// with full-rank E. This is the continuous floor after drift.
    pub fn projector_matrix_tikhonov(&self, lambda: f64) -> AmbientMatrix {
        let et = self.e.transpose();
        let gram: MercyGram = et * &self.e;
        let lam = lambda.max(0.0);
        let damped = gram + MercyGram::identity() * lam;
        match damped.try_inverse() {
            Some(ginv) => &self.e * ginv * et,
            None => self.projector_matrix_exact(),
        }
    }
    pub fn projector_matrix(&self) -> AmbientMatrix {
        if self.tikhonov_lambda <= 0.0 {
            self.projector_matrix_exact()
        } else {
            self.projector_matrix_tikhonov(self.tikhonov_lambda)
        }
    }
    pub fn ambient_dim(&self) -> usize { AMBIENT_DIM }
    pub fn mercy_dim(&self) -> usize { MERCY_DIM }
}
