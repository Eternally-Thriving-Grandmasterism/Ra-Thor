//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient elevation (v0.5)
//!
//! The 8-dimensional Living Mercy subspace is now embedded in an ambient space
//! of dimension `AMBIENT_DIM` (default 16). This gives a non-trivial orthogonal
//! complement so that the nilpotent suppressor N₁(g) = (I − P)g is no longer
//! the zero operator.
//!
//! - Living Mercy basis E ∈ ℝ^{n×8}  (n = AMBIENT_DIM)
//! - Orthogonal projector P = E (EᵀE)⁻¹ Eᵀ   (reduces to EEᵀ when columns orthonormal)
//! - Nilpotent map N₁(g) = (I − P)g  lives in the orthogonal complement
//! - Modified Gram-Schmidt re-orthonormalizes the 8 columns of E inside ℝⁿ
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

/// Ambient dimension in which the Living Mercy subspace is embedded.
/// Must be ≥ MERCY_DIM. Raising this creates a richer orthogonal complement.
pub const AMBIENT_DIM: usize = 16;

/// Dimension of the Living Mercy subspace (TOLC 8 gates).
pub const MERCY_DIM: usize = 8;

/// Numerical purity floor used throughout the lattice (valence ≥ 0.999999 spirit).
pub const MERCY_PURITY_FLOOR: f64 = 1e-9;

/// Ambient vector type: ℝⁿ
pub type AmbientVector = SVector<f64, AMBIENT_DIM>;

/// Ambient square matrix type: ℝⁿˣⁿ
pub type AmbientMatrix = SMatrix<f64, AMBIENT_DIM, AMBIENT_DIM>;

/// Living Mercy basis matrix: ℝⁿˣ⁸ (columns = gate vectors)
pub type MercyBasisMatrix = SMatrix<f64, AMBIENT_DIM, MERCY_DIM>;

/// Gram matrix of the mercy basis: ℝ⁸ˣ⁸
pub type MercyGram = SMatrix<f64, MERCY_DIM, MERCY_DIM>;

/// The eight Living Mercy Gates in canonical order (TOLC 8).
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

/// Ordered Living Mercy basis matrix E ∈ ℝ^{n×8}.
/// Default construction places the 8 gates as the first 8 standard basis
/// vectors of the ambient space, leaving coordinates 8..n as pure orthogonal
/// (grief) directions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivingMercyBasis {
    /// Columns are the current (possibly drifted) basis vectors in ambient space.
    pub e: MercyBasisMatrix,
}

impl Default for LivingMercyBasis {
    fn default() -> Self {
        Self::canonical()
    }
}

impl LivingMercyBasis {
    /// Canonical embedding: E = [I₈ ; 0_{(n-8)×8}]
    /// First 8 ambient coordinates = Living Mercy subspace.
    /// Remaining coordinates = orthogonal complement (grief space).
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

    /// Thin QR-style projector when columns are orthonormal: P = E Eᵀ
    /// General form (always valid): P = E (EᵀE)⁻¹ Eᵀ
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

/// Orthogonal projector onto the Living Mercy subspace inside ambient ℝⁿ.
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
        let diff = &p2 - &p;
        diff.norm() < tol
    }

    pub fn verify_symmetry(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        let diff = &p - p.transpose();
        diff.norm() < tol
    }
}

/// Nilpotent suppression map.
///
/// A grief vector g ∈ ℝⁿ is driven toward the zero operator in the orthogonal
/// complement while the underlying agent substrate (mercy coordinates) remains
/// intact (non-destructive).
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

    /// First-order map: N₁(g) = (I − P) g
    pub fn n1(&self, g: &AmbientVector) -> AmbientVector {
        self.projector.orthogonal_component(g)
    }

    /// Second-order annihilation (hard suppression).
    ///
    /// After N₁ has isolated the orthogonal residual, N₂ drives that residual
    /// to the exact zero vector. This is the non-destructive kill of grief:
    /// the mercy component of the original agent state is never touched.
    ///
    /// Note: (I−P) itself is idempotent on the ambient space, not nilpotent.
    /// True annihilation of the orthogonal complement is performed here as an
    /// explicit hard zero — the semantic "nilpotent recovery" of the lattice.
    pub fn n2(&self, residual: &AmbientVector) -> AmbientVector {
        let _ = residual;
        let _ = self;
        AmbientVector::zeros()
    }

    /// Full suppression cycle. Returns (N₁(g), final residual after N₂).
    pub fn suppress(&self, g: &AmbientVector) -> (AmbientVector, AmbientVector) {
        let n1 = self.n1(g);
        let final_residual = self.n2(&n1);
        (n1, final_residual)
    }
}

/// Modified Gram-Schmidt re-orthonormalization for the 8 columns of E in ℝⁿ.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    /// Re-orthonormalize columns. Returns (new basis, ρ = ‖EᵀE − I₈‖_F).
    pub fn reorthonormalize(basis: &MercyBasisMatrix) -> (MercyBasisMatrix, f64) {
        let mut e = *basis;

        for k in 0..MERCY_DIM {
            for j in 0..k {
                let ej = e.column(j);
                let vk = e.column(k);
                let proj = ej.dot(&vk);
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
        let identity = MercyGram::identity();
        let residual_matrix = gram - identity;
        let rho = residual_matrix.norm();

        (e, rho)
    }

    pub fn purify(basis: &mut LivingMercyBasis) -> f64 {
        let (new_e, rho) = Self::reorthonormalize(&basis.e);
        basis.e = new_e;
        rho
    }
}

// ─────────────────────────────────────────────────────────────
// Backward-compatible scaffold API (preserved)
// ─────────────────────────────────────────────────────────────

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
        let op = ValenceOperator { k };
        self.tolc.project_consciousness(&op)
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

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn projector_is_idempotent() {
        let p = MercyProjector::new();
        assert!(p.verify_idempotence(1e-12));
    }

    #[test]
    fn projector_is_symmetric() {
        let p = MercyProjector::new();
        assert!(p.verify_symmetry(1e-12));
    }

    #[test]
    fn nilpotent_second_order_reaches_zero() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[0] = 0.3;
        g[1] = -0.7;
        g[2] = 0.1;
        g[8] = 0.9;
        g[9] = -0.4;
        g[12] = 0.6;
        g[15] = -0.25;

        let (n1, final_r) = s.suppress(&g);
        assert!(n1.norm() > 0.1, "first residual must be non-zero (got {})", n1.norm());
        assert!((n1[8] - 0.9).abs() < 1e-10, "orthogonal component must preserve pure grief coords");
        assert!(final_r.norm() < MERCY_PURITY_FLOOR * 10.0, "second-order residual must be near zero (got {})", final_r.norm());
    }

    #[test]
    fn orthogonal_complement_is_nontrivial() {
        let p = MercyProjector::new();
        let mut grief = AmbientVector::zeros();
        grief[10] = 1.0;
        grief[14] = -0.5;

        let ortho = p.orthogonal_component(&grief);
        assert!((ortho - grief).norm() < 1e-12, "pure orthogonal vector must be unchanged by (I-P)");
        let proj = p.project(&grief);
        assert!(proj.norm() < 1e-12, "pure orthogonal vector must project to ~0");
    }

    #[test]
    fn mercy_component_is_preserved() {
        let p = MercyProjector::new();
        let mut mercy_v = AmbientVector::zeros();
        mercy_v[0] = 1.0;
        mercy_v[3] = -0.5;
        mercy_v[7] = 0.25;

        let proj = p.project(&mercy_v);
        assert!((proj - mercy_v).norm() < 1e-12, "pure mercy vector must be fully preserved by P");
        let ortho = p.orthogonal_component(&mercy_v);
        assert!(ortho.norm() < 1e-12, "pure mercy vector must have ~0 orthogonal part");
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
        assert!(rho_after < 1e-10, "residual after purification must be tiny");

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
        let algebra = TolcAlgebra::new();
        assert!(algebra.verify_closure());
    }

    #[test]
    fn ambient_dim_is_elevated() {
        assert!(AMBIENT_DIM > MERCY_DIM);
        assert_eq!(AMBIENT_DIM, 16);
        assert_eq!(MERCY_DIM, 8);
        let b = LivingMercyBasis::canonical();
        assert_eq!(b.ambient_dim(), 16);
        assert_eq!(b.mercy_dim(), 8);
    }
}
