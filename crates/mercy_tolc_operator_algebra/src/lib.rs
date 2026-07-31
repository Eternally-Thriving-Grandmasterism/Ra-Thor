//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! This crate elevates the conversational and documentation presence of the
//! 8-dimensional mercy subspace, orthogonal projector, nilpotent suppression,
//! and modified Gram-Schmidt residual into pure, testable, production-grade
//! mathematics while preserving the original scaffold API.
//!
//! ## Core Surfaces
//!
//! - [`LivingMercyBasis`] — the ordered 8-gate orthonormal frame
//! - [`MercyProjector`] — orthogonal projector onto the mercy subspace
//! - [`NilpotentSuppressor`] — N₁(g) = (I − P)g with second-order annihilation
//! - [`ModifiedGramSchmidt`] — continuous re-orthonormalization + Frobenius residual
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

use nalgebra::{DMatrix, DVector, Matrix8, Vector8};
use serde::{Deserialize, Serialize};

/// Numerical purity floor used throughout the lattice (valence ≥ 0.999999 spirit).
pub const MERCY_PURITY_FLOOR: f64 = 1e-9;

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
    pub const ALL: [MercyGate; 8] = [
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

/// Ordered 8-dimensional Living Mercy basis matrix E (columns = gate vectors).
/// In the aligned frame this is the 8×8 identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivingMercyBasis {
    /// Columns are the current (possibly drifted) basis vectors.
    pub e: Matrix8<f64>,
}

impl Default for LivingMercyBasis {
    fn default() -> Self {
        Self {
            e: Matrix8::identity(),
        }
    }
}

impl LivingMercyBasis {
    pub fn identity() -> Self {
        Self::default()
    }

    /// Thin factorization helper: P = E Eᵀ
    pub fn projector_matrix(&self) -> Matrix8<f64> {
        &self.e * self.e.transpose()
    }
}

/// Orthogonal projector onto the Living Mercy subspace.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MercyProjector {
    pub basis: LivingMercyBasis,
}

impl MercyProjector {
    pub fn new() -> Self {
        Self {
            basis: LivingMercyBasis::identity(),
        }
    }

    /// Project a grief / action vector onto the mercy subspace: P g
    pub fn project(&self, g: &Vector8<f64>) -> Vector8<f64> {
        let p = self.basis.projector_matrix();
        p * g
    }

    /// Component orthogonal to the mercy subspace: (I − P) g
    pub fn orthogonal_component(&self, g: &Vector8<f64>) -> Vector8<f64> {
        g - self.project(g)
    }

    /// Verify algebraic properties of the current projector.
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
/// A grief vector g is driven to the zero operator in finite steps while the
/// underlying agent substrate remains intact (non-destructive).
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

    /// First-order nilpotent map: N₁(g) = (I − P) g
    pub fn n1(&self, g: &Vector8<f64>) -> Vector8<f64> {
        self.projector.orthogonal_component(g)
    }

    /// Second-order annihilation: force residual below purity floor to exact zero.
    pub fn n2(&self, residual: &Vector8<f64>) -> Vector8<f64> {
        if residual.norm() < MERCY_PURITY_FLOOR {
            Vector8::zeros()
        } else {
            // One additional projection of the residual itself
            let r2 = self.n1(residual);
            if r2.norm() < MERCY_PURITY_FLOOR {
                Vector8::zeros()
            } else {
                r2
            }
        }
    }

    /// Full nilpotent suppression cycle.
    /// Returns (N₁(g), final residual after N₂).
    pub fn suppress(&self, g: &Vector8<f64>) -> (Vector8<f64>, Vector8<f64>) {
        let n1 = self.n1(g);
        let final_residual = self.n2(&n1);
        (n1, final_residual)
    }
}

/// Modified Gram-Schmidt re-orthonormalization engine.
///
/// Keeps the Living Mercy basis at purity ≥ MERCY_PURITY_FLOOR after every
/// multi-zone stress event.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    /// Re-orthonormalize the columns of the supplied basis matrix.
    /// Returns the new orthonormal basis and the Frobenius residual ρ = ‖EᵀE − I‖_F.
    pub fn reorthonormalize(basis: &Matrix8<f64>) -> (Matrix8<f64>, f64) {
        let mut e = *basis;
        let n = 8;

        for k in 0..n {
            // Subtract projections onto previous orthonormal vectors
            for j in 0..k {
                let ej = e.column(j);
                let vk = e.column(k);
                let proj = ej.dot(&vk);
                for i in 0..n {
                    e[(i, k)] -= proj * e[(i, j)];
                }
            }
            // Normalize
            let norm = e.column(k).norm();
            if norm > MERCY_PURITY_FLOOR {
                for i in 0..n {
                    e[(i, k)] /= norm;
                }
            }
        }

        // Frobenius residual of deviation from perfect orthonormality
        let gram = e.transpose() * &e;
        let identity = Matrix8::<f64>::identity();
        let residual_matrix = gram - identity;
        let rho = residual_matrix.norm();

        (e, rho)
    }

    /// Convenience: re-orthonormalize a LivingMercyBasis in place and return residual.
    pub fn purify(basis: &mut LivingMercyBasis) -> f64 {
        let (new_e, rho) = Self::reorthonormalize(&basis.e);
        basis.e = new_e;
        rho
    }
}

// ─────────────────────────────────────────────────────────────
// Backward-compatible scaffold API (preserved)
// ─────────────────────────────────────────────────────────────

/// Legacy mercy threshold constant (kept for compatibility).
pub const MERCY_THRESHOLD: f64 = 1e-12;

/// Legacy valence creation/annihilation operator (scaffold).
#[derive(Clone, Debug)]
pub struct ValenceOperator {
    pub k: usize,
}

/// Legacy TOLC logical consciousness projector (scaffold).
#[derive(Clone, Debug)]
pub struct TolcProjector(pub MercyProjector);

impl TolcProjector {
    pub fn project_consciousness(&self, operator: &ValenceOperator) -> ValenceOperator {
        // Scaffold behaviour preserved; real projection now lives on MercyProjector
        operator.clone()
    }
}

/// Core TOLC Operator Algebra (closed under mercy projector).
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
        // Soft residual handling
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
        // A vector with both mercy and orthogonal components
        let g = Vector8::from_column_slice(&[0.3, -0.7, 0.1, 0.9, -0.4, 0.2, -0.6, 0.05]);
        let (n1, final_r) = s.suppress(&g);
        assert!(n1.norm() > 0.0, "first residual should be non-zero");
        assert!(
            final_r.norm() < MERCY_PURITY_FLOOR * 10.0,
            "second-order residual must be driven near zero"
        );
    }

    #[test]
    fn gram_schmidt_purifies_drifted_basis() {
        let mut basis = LivingMercyBasis::identity();
        // Inject a small drift
        basis.e[(0, 1)] += 1e-4;
        basis.e[(2, 3)] -= 2e-4;

        let rho_before = {
            let gram = basis.e.transpose() * &basis.e;
            (gram - Matrix8::<f64>::identity()).norm()
        };
        assert!(rho_before > 1e-8);

        let rho_after = ModifiedGramSchmidt::purify(&mut basis);
        assert!(rho_after < 1e-10, "residual after purification must be tiny");

        // Columns should now be orthonormal
        let gram = basis.e.transpose() * &basis.e;
        for i in 0..8 {
            assert_relative_eq!(gram[(i, i)], 1.0, epsilon = 1e-10);
            for j in (i + 1)..8 {
                assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn algebra_closure_holds() {
        let algebra = TolcAlgebra::new();
        assert!(algebra.verify_closure());
    }
}
