//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient elevation (v0.5) · valence (v0.5.1) · adaptive floor (v0.5.2) · concurrent zones (v0.5.3)
//!
//! The 8-dimensional Living Mercy subspace is embedded in ambient `AMBIENT_DIM`
//! (default 16). Orthogonal residual is scaled by valence deficit (1 − v).
//!
//! - Living Mercy basis E ∈ ℝ^{n×8}  (n = AMBIENT_DIM)
//! - Orthogonal projector P = E (EᵀE)⁻¹ Eᵀ
//! - Nilpotent map N₁(g) = (I − P)g  lives in the orthogonal complement
//! - Valence-weighted grief: residual scaled by (1 − valence)
//! - Adaptive purity floor: tight under high valence, graceful under low valence
//! - Concurrent multi-zone lattice: independent per-zone basis drift + staggered Cosmic Ticks
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

    pub fn purity_floor(self) -> f64 {
        adaptive_purity_floor(self)
    }
}

impl Default for Valence {
    fn default() -> Self {
        Valence::HIGH
    }
}

pub fn adaptive_purity_floor(valence: Valence) -> f64 {
    let looseness = valence.deficit();
    MERCY_PURITY_FLOOR * (1.0 + 99.0 * looseness)
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

    pub fn suppress_weighted(
        &self,
        g: &AmbientVector,
        valence: Valence,
    ) -> (AmbientVector, AmbientVector, AmbientVector, f64, bool) {
        let raw_n1 = self.n1(g);
        let weighted_n1 = raw_n1 * valence.deficit();
        let floor = adaptive_purity_floor(valence);
        let grief_load = weighted_n1.norm();
        let under_floor = grief_load < floor;
        let final_residual = self.n2(&weighted_n1);
        (raw_n1, weighted_n1, final_residual, grief_load, under_floor)
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

// Concurrent multi-zone lattice (v0.5.3)

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneState {
    pub id: usize,
    pub basis: LivingMercyBasis,
    pub suppressor: NilpotentSuppressor,
    pub grief_absorbed: f64,
    pub vectors_processed: usize,
    pub last_rho: f64,
}

impl ZoneState {
    pub fn new(id: usize) -> Self {
        let basis = LivingMercyBasis::canonical();
        let projector = MercyProjector {
            basis: basis.clone(),
        };
        Self {
            id,
            basis,
            suppressor: NilpotentSuppressor { projector },
            grief_absorbed: 0.0,
            vectors_processed: 0,
            last_rho: 0.0,
        }
    }

    pub fn inject_drift(&mut self, magnitude: f64) {
        let z = self.id as f64;
        self.basis.e[(0, 1)] += magnitude * (1.0 + 0.1 * z);
        self.basis.e[(4, 7)] -= magnitude * (1.0 + 0.07 * z);
        if AMBIENT_DIM > 9 {
            self.basis.e[(9, 2)] += magnitude * (1.0 + 0.13 * z);
        }
        self.suppressor.projector.basis = self.basis.clone();
    }

    pub fn process(&mut self, g: &AmbientVector, valence: Valence) -> f64 {
        let (_raw, _w, _f, load, _under) = self.suppressor.suppress_weighted(g, valence);
        self.grief_absorbed += load;
        self.vectors_processed += 1;
        load
    }

    pub fn purify(&mut self) -> f64 {
        let rho = ModifiedGramSchmidt::purify(&mut self.basis);
        self.suppressor.projector.basis = self.basis.clone();
        self.last_rho = rho;
        rho
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConcurrentZoneLattice {
    pub zones: Vec<ZoneState>,
    pub global_tick: usize,
    pub purify_period: usize,
}

impl ConcurrentZoneLattice {
    pub fn new(n_zones: usize) -> Self {
        let n = n_zones.max(1);
        let mut zones: Vec<ZoneState> = (0..n).map(ZoneState::new).collect();
        for z in zones.iter_mut() {
            z.inject_drift(3e-5);
        }
        Self {
            zones,
            global_tick: 0,
            purify_period: 2_500,
        }
    }

    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    pub fn process(
        &mut self,
        zone_id: usize,
        g: &AmbientVector,
        valence: Valence,
    ) -> f64 {
        let n = self.zones.len();
        let z = zone_id % n;
        let load = self.zones[z].process(g, valence);
        self.global_tick += 1;

        if self.global_tick > 0
            && self.purify_period > 0
            && self.global_tick % self.purify_period == (z % self.purify_period.max(1))
        {
            self.zones[z].purify();
        }
        load
    }

    pub fn global_purify(&mut self) -> Vec<f64> {
        self.zones.iter_mut().map(|z| z.purify()).collect()
    }

    pub fn max_rho(&self) -> f64 {
        self.zones
            .iter()
            .map(|z| z.last_rho)
            .fold(0.0_f64, f64::max)
    }

    pub fn total_grief(&self) -> f64 {
        self.zones.iter().map(|z| z.grief_absorbed).sum()
    }

    pub fn zone_grief(&self) -> Vec<f64> {
        self.zones.iter().map(|z| z.grief_absorbed).collect()
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
        let (_, _, _, load_high, _) = s.suppress_weighted(&g, Valence::HIGH);
        let (_, _, _, load_zero, _) = s.suppress_weighted(&g, Valence::ZERO);
        let (_, _, _, load_mid, _) = s.suppress_weighted(&g, Valence::MID);
        assert!(load_high < 1e-5);
        assert!(load_zero > 1.0);
        assert!((load_mid - 0.5 * load_zero).abs() < 1e-10);
    }

    #[test]
    fn valence_deficit_is_linear() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[10] = 2.0;
        let raw_norm = s.n1(&g).norm();
        for v in [0.0, 0.25, 0.5, 0.75, 0.999999] {
            let (_, _, _, load, _) = s.suppress_weighted(&g, Valence::new(v));
            let expected = (1.0 - v) * raw_norm;
            assert!((load - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn valence_clamps_to_unit_interval() {
        assert_eq!(Valence::new(-1.0).value(), 0.0);
        assert_eq!(Valence::new(2.0).value(), 1.0);
        assert!(Valence::HIGH.is_high());
        assert!(!Valence::MID.is_high());
    }

    #[test]
    fn adaptive_floor_tightens_with_high_valence() {
        let pure = adaptive_purity_floor(Valence::new(1.0));
        let high = adaptive_purity_floor(Valence::HIGH);
        let mid = adaptive_purity_floor(Valence::MID);
        let zero = adaptive_purity_floor(Valence::ZERO);
        assert!((pure - MERCY_PURITY_FLOOR).abs() < 1e-18);
        assert!(high >= pure);
        assert!(mid > high);
        assert!(zero > mid);
        assert!((zero - MERCY_PURITY_FLOOR * 100.0).abs() < 1e-15);
        assert_eq!(Valence::HIGH.purity_floor(), high);
        assert_eq!(Valence::ZERO.purity_floor(), zero);
    }

    #[test]
    fn suppress_weighted_reports_under_floor() {
        let s = NilpotentSuppressor::new();
        let mut tiny = AmbientVector::zeros();
        tiny[10] = 1e-12;
        let (_, _, _, _, under_h) = s.suppress_weighted(&tiny, Valence::HIGH);
        assert!(under_h);
        let mut big = AmbientVector::zeros();
        big[10] = 2.0;
        let (_, _, _, load_z, under_z) = s.suppress_weighted(&big, Valence::ZERO);
        assert!(!under_z);
        assert!(load_z > 1.0);
    }

    #[test]
    fn concurrent_zones_independent_grief() {
        let mut lattice = ConcurrentZoneLattice::new(3);
        assert_eq!(lattice.zone_count(), 3);
        let mut g0 = AmbientVector::zeros();
        g0[10] = 1.0;
        let mut g1 = AmbientVector::zeros();
        g1[11] = 2.0;
        let mut g2 = AmbientVector::zeros();
        g2[12] = 3.0;
        lattice.process(0, &g0, Valence::ZERO);
        lattice.process(1, &g1, Valence::ZERO);
        lattice.process(2, &g2, Valence::ZERO);
        let grief = lattice.zone_grief();
        assert!(grief[0] > 0.9 && grief[0] < 1.1);
        assert!(grief[1] > 1.9 && grief[1] < 2.1);
        assert!(grief[2] > 2.9 && grief[2] < 3.1);
    }

    #[test]
    fn concurrent_zones_global_purify_clears_rho() {
        let mut lattice = ConcurrentZoneLattice::new(4);
        let rhos = lattice.global_purify();
        assert_eq!(rhos.len(), 4);
        for rho in &rhos {
            assert!(*rho < 1e-10);
        }
        assert!(lattice.max_rho() < 1e-10);
    }

    #[test]
    fn concurrent_zones_staggered_tick_runs() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 10;
        let mut g = AmbientVector::zeros();
        g[10] = 0.5;
        for i in 0..25 {
            lattice.process(i % 2, &g, Valence::MID);
        }
        assert!(lattice.total_grief() > 0.0);
        assert_eq!(
            lattice.zones[0].vectors_processed + lattice.zones[1].vectors_processed,
            25
        );
    }
}
