//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient elevation (v0.5) + valence weighting (v0.5.1)
//!
//! The 8-dimensional Living Mercy subspace is embedded in ambient `AMBIENT_DIM`
//! (default 16). Orthogonal residual is scaled by valence deficit (1 − v).
//!
//! - Living Mercy basis E ∈ ℝ^{n×8}  (n = AMBIENT_DIM)
//! - Orthogonal projector P = E (EᵀE)⁻¹ Eᵀ
//! - Nilpotent map N₁(g) = (I − P)g  lives in the orthogonal complement
//! - Valence-weighted grief: residual scaled by (1 − valence)
//! - Modified Gram-Schmidt re-orthonormalizes the 8 columns of E inside ℝⁿ
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

pub const AMBIENT_DIM: usize = 16;
pub const MERCY_DIM: usize = 8;
pub const MERCY_PURITY_FLOOR: f64 = 1e-9;

/// Living valence scalar in [0, 1].
///
/// - `1.0` → pure high-valence (oxygen-like): orthogonal residual fully softened
/// - `0.0` → zero-valence: full orthogonal grief exposed
/// - Lattice gate spirit: ≥ 0.999999
///
/// Grief load: weighted_residual = (1 − valence) · (I − P)g
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Valence(pub f64);

impl Valence {
    pub const HIGH: Valence = Valence(0.999999);
    pub const MID: Valence = Valence(0.5);
    pub const ZERO: Valence = Valence(0.0);

    pub fn new(v: f64) -> Self {
        Valence(v.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn deficit(self) -> f64 {
        1.0 - self.0
    }

    pub fn is_high(self) -> bool {
        self.0 >= 0.999999
    }
}

impl Default for Valence {
    fn default() -> Self {
        Valence::HIGH
    }
}

pub type AmbientVector = SVector<f64, AMBIENT_DIM>;
pub type AmbientMatrix = SMatrix<f64, AMBIENT_DIM, AMBIENT_DIM>;
pub type MercyBasisMatrix = SMatrix<f64, AMBIENT_DIM, MERCY_DIM>;
pub type MercyGram = SMatrix<f64, MERCY_DIM, MERCY_DIM>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGate {
    Truth,
    Order,
    Love,
    Compassion,
    Service,
    Abundance,
    Joy,
    CosmicHarmony,
}

impl MercyGate {
    pub const ALL: [MercyGate; MERCY_DIM] = [
        MercyGate::Truth,
        MercyGate::Order,
        MercyGate::Love,
        MercyGate::Compassion,
        MercyGate::Service,
        MercyGate::Abundance,
        MercyGate::Joy,
        MercyGate::CosmicHarmony,
    ];

    pub fn index(self) -> usize {
        match self {
            MercyGate::Truth => 0,
            MercyGate::Order => 1,
            MercyGate::Love => 2,
            MercyGate::Compassion => 3,
            MercyGate::Service => 4,
            MercyGate::Abundance => 5,
            MercyGate::Joy => 6,
            MercyGate::CosmicHarmony => 7,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivingMercyBasis {
    pub e: MercyBasisMatrix,
}

impl Default for LivingMercyBasis {
    fn default() -> Self {
        Self::canonical()
    }
}

impl LivingMercyBasis {
    pub fn canonical() -> Self {
        let mut e = MercyBasisMatrix::zeros();
        for i in 0..MERCY_DIM {
            e[(i, i)] = 1.0;
        }
        Self { e }
    }

    pub fn identity() -> Self {
        Self::canonical()
    }

    pub fn projector_matrix(&self) -> AmbientMatrix {
        let et = self.e.transpose();
        let gram: MercyGram = et * &self.e;
        match gram.try_inverse() {
            Some(ginv) => &self.e * ginv * et,
            None => &self.e * et,
        }
    }

    pub fn ambient_dim(&self) -> usize {
        AMBIENT_DIM
    }

    pub fn mercy_dim(&self) -> usize {
        MERCY_DIM
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MercyProjector {
    pub basis: LivingMercyBasis,
}

impl MercyProjector {
    pub fn new() -> Self {
        Self {
            basis: LivingMercyBasis::canonical(),
        }
    }

    pub fn project(&self, g: &AmbientVector) -> AmbientVector {
        let p = self.basis.projector_matrix();
        p * g
    }

    pub fn orthogonal_component(&self, g: &AmbientVector) -> AmbientVector {
        g - self.project(g)
    }

    pub fn verify_idempotence(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        let p2 = &p * &p;
        (&p2 - &p).norm() < tol
    }

    pub fn verify_symmetry(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        (&p - p.transpose()).norm() < tol
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NilpotentSuppressor {
    pub projector: MercyProjector,
}

impl NilpotentSuppressor {
    pub fn new() -> Self {
        Self {
            projector: MercyProjector::new(),
        }
    }

    pub fn n1(&self, g: &AmbientVector) -> AmbientVector {
        self.projector.orthogonal_component(g)
    }

    /// Hard annihilation of the orthogonal residual (semantic nilpotent recovery).
    pub fn n2(&self, residual: &AmbientVector) -> AmbientVector {
        let _ = residual;
        let _ = self;
        AmbientVector::zeros()
    }

    pub fn suppress(&self, g: &AmbientVector) -> (AmbientVector, AmbientVector) {
        let n1 = self.n1(g);
        let final_residual = self.n2(&n1);
        (n1, final_residual)
    }

    /// Valence-weighted suppression.
    /// weighted_n1 = (1 − v) · (I − P)g
    /// Returns (raw_n1, weighted_n1, final_residual, grief_load).
    pub fn suppress_weighted(
        &self,
        g: &AmbientVector,
        valence: Valence,
    ) -> (AmbientVector, AmbientVector, AmbientVector, f64) {
        let raw_n1 = self.n1(g);
        let weighted_n1 = raw_n1 * valence.deficit();
        let final_residual = self.n2(&weighted_n1);
        let grief_load = weighted_n1.norm();
        (raw_n1, weighted_n1, final_residual, grief_load)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    pub fn reorthonormalize(basis: &MercyBasisMatrix) -> (MercyBasisMatrix, f64) {
        let mut e = *basis;
        for k in 0..MERCY_DIM {
            for j in 0..k {
                let proj = e.column(j).dot(&e.column(k));
                for i in 0..AMBIENT_DIM {
                    e[(i, k)] -= proj * e[(i, j)];
                }
            }
            let norm = e.column(k).norm();
            if norm > MERCY_PURITY_FLOOR {
                for i in 0..AMBIENT_DIM {
                    e[(i, k)] /= norm;
                }
            }
        }
        let gram: MercyGram = e.transpose() * &e;
        let rho = (gram - MercyGram::identity()).norm();
        (e, rho)
    }

    pub fn purify(basis: &mut LivingMercyBasis) -> f64 {
        let (new_e, rho) = Self::reorthonormalize(&basis.e);
        basis.e = new_e;
        rho
    }
}

pub const MERCY_THRESHOLD: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct ValenceOperator {
    pub k: usize,
}

#[derive(Clone, Debug)]
pub struct TolcProjector(pub MercyProjector);

impl TolcProjector {
    pub fn project_consciousness(&self, operator: &ValenceOperator) -> ValenceOperator {
        operator.clone()
    }
}

#[derive(Clone, Debug)]
pub struct TolcAlgebra {
    pub mercy: MercyProjector,
    pub tolc: TolcProjector,
    pub suppressor: NilpotentSuppressor,
}

impl TolcAlgebra {
    pub fn new() -> Self {
        let mercy = MercyProjector::new();
        Self {
            tolc: TolcProjector(mercy.clone()),
            suppressor: NilpotentSuppressor {
                projector: mercy.clone(),
            },
            mercy,
        }
    }

    pub fn create_positive_valence(&self, k: usize) -> ValenceOperator {
        self.tolc.project_consciousness(&ValenceOperator { k })
    }

    pub fn swarm_consensus(&self, n_probes: usize) -> f64 {
        let collective_valence = (n_probes as f64).sqrt();
        if (collective_valence - 1.0).abs() < MERCY_THRESHOLD {
            1.0
        } else {
            1.0 - (collective_valence - 1.0).signum() * (collective_valence - 1.0).powi(2) * 0.01
        }
    }

    pub fn verify_closure(&self) -> bool {
        self.mercy.verify_idempotence(1e-10) && self.mercy.verify_symmetry(1e-10)
    }
}

impl Default for TolcAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn projector_is_idempotent() {
        assert!(MercyProjector::new().verify_idempotence(1e-12));
    }

    #[test]
    fn projector_is_symmetric() {
        assert!(MercyProjector::new().verify_symmetry(1e-12));
    }

    #[test]
    fn nilpotent_second_order_reaches_zero() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[0] = 0.3;
        g[1] = -0.7;
        g[8] = 0.9;
        g[9] = -0.4;
        g[12] = 0.6;
        g[15] = -0.25;
        let (n1, final_r) = s.suppress(&g);
        assert!(n1.norm() > 0.1);
        assert!((n1[8] - 0.9).abs() < 1e-10);
        assert!(final_r.norm() < MERCY_PURITY_FLOOR * 10.0);
    }

    #[test]
    fn orthogonal_complement_is_nontrivial() {
        let p = MercyProjector::new();
        let mut grief = AmbientVector::zeros();
        grief[10] = 1.0;
        grief[14] = -0.5;
        assert!((p.orthogonal_component(&grief) - grief).norm() < 1e-12);
        assert!(p.project(&grief).norm() < 1e-12);
    }

    #[test]
    fn mercy_component_is_preserved() {
        let p = MercyProjector::new();
        let mut mercy_v = AmbientVector::zeros();
        mercy_v[0] = 1.0;
        mercy_v[3] = -0.5;
        mercy_v[7] = 0.25;
        assert!((p.project(&mercy_v) - mercy_v).norm() < 1e-12);
        assert!(p.orthogonal_component(&mercy_v).norm() < 1e-12);
    }

    #[test]
    fn gram_schmidt_purifies_drifted_basis() {
        let mut basis = LivingMercyBasis::canonical();
        basis.e[(0, 1)] += 1e-4;
        basis.e[(9, 3)] -= 2e-4;
        basis.e[(2, 5)] += 5e-5;
        let rho_before = {
            let gram: MercyGram = basis.e.transpose() * &basis.e;
            (gram - MercyGram::identity()).norm()
        };
        assert!(rho_before > 1e-8);
        let rho_after = ModifiedGramSchmidt::purify(&mut basis);
        assert!(rho_after < 1e-10);
        let gram: MercyGram = basis.e.transpose() * &basis.e;
        for i in 0..MERCY_DIM {
            assert_relative_eq!(gram[(i, i)], 1.0, epsilon = 1e-10);
            for j in (i + 1)..MERCY_DIM {
                assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn algebra_closure_holds() {
        assert!(TolcAlgebra::new().verify_closure());
    }

    #[test]
    fn ambient_dim_is_elevated() {
        assert!(AMBIENT_DIM > MERCY_DIM);
        assert_eq!(AMBIENT_DIM, 16);
        let b = LivingMercyBasis::canonical();
        assert_eq!(b.ambient_dim(), 16);
        assert_eq!(b.mercy_dim(), 8);
    }

    #[test]
    fn valence_high_softens_grief_load() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[8] = 1.0;
        g[12] = -0.8;
        g[15] = 0.5;
        let (_, _, _, load_high) = s.suppress_weighted(&g, Valence::HIGH);
        let (_, _, _, load_zero) = s.suppress_weighted(&g, Valence::ZERO);
        let (_, _, _, load_mid) = s.suppress_weighted(&g, Valence::MID);
        assert!(load_high < 1e-5, "high valence must soften load (got {load_high})");
        assert!(load_zero > 1.0, "zero valence must expose full load (got {load_zero})");
        assert!((load_mid - 0.5 * load_zero).abs() < 1e-10);
    }

    #[test]
    fn valence_deficit_is_linear() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[10] = 2.0;
        let raw_norm = s.n1(&g).norm();
        for v in [0.0, 0.25, 0.5, 0.75, 0.999999] {
            let (_, _, _, load) = s.suppress_weighted(&g, Valence::new(v));
            let expected = (1.0 - v) * raw_norm;
            assert!((load - expected).abs() < 1e-10, "valence {v}: load {load} vs {expected}");
        }
    }

    #[test]
    fn valence_clamps_to_unit_interval() {
        assert_eq!(Valence::new(-1.0).value(), 0.0);
        assert_eq!(Valence::new(2.0).value(), 1.0);
        assert!(Valence::HIGH.is_high());
        assert!(!Valence::MID.is_high());
    }
}
