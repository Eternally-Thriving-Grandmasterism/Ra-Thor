use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

pub const AMBIENT_DIM: usize = 16;
pub const MERCY_DIM: usize = 8;
pub const MERCY_PURITY_FLOOR: f64 = 1e-9;
pub const TIKHONOV_RHO_GAIN: f64 = 1.0;
pub const TIKHONOV_STRESS_GAIN: f64 = 1e-6;

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
    pub fn projector_matrix_exact(&self) -> AmbientMatrix {
        let et = self.e.transpose();
        let gram: MercyGram = et * &self.e;
        match gram.try_inverse() {
            Some(ginv) => &self.e * ginv * et,
            None => &self.e * et,
        }
    }
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MercyProjector { pub basis: LivingMercyBasis }

impl MercyProjector {
    pub fn new() -> Self { Self { basis: LivingMercyBasis::canonical() } }
    pub fn project(&self, g: &AmbientVector) -> AmbientVector { self.basis.projector_matrix() * g }
    pub fn orthogonal_component(&self, g: &AmbientVector) -> AmbientVector { g - self.project(g) }
    pub fn verify_idempotence(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        (&p * &p - &p).norm() < tol
    }
    pub fn verify_symmetry(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        (&p - p.transpose()).norm() < tol
    }
    pub fn verify_symmetric_psd(&self, tol: f64) -> bool {
        let p = self.basis.projector_matrix();
        if (&p - p.transpose()).norm() >= tol { return false; }
        let eig = nalgebra::SymmetricEigen::new((p + p.transpose()) * 0.5);
        eig.eigenvalues.iter().all(|ev| *ev >= -tol)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NilpotentSuppressor { pub projector: MercyProjector }

impl NilpotentSuppressor {
    pub fn new() -> Self { Self { projector: MercyProjector::new() } }
    pub fn n1(&self, g: &AmbientVector) -> AmbientVector { self.projector.orthogonal_component(g) }
    pub fn n2(&self, residual: &AmbientVector) -> AmbientVector {
        let _ = (residual, self); AmbientVector::zeros()
    }
    pub fn suppress(&self, g: &AmbientVector) -> (AmbientVector, AmbientVector) {
        let n1 = self.n1(g); (n1, self.n2(&n1))
    }
    pub fn suppress_weighted(
        &self, g: &AmbientVector, valence: Valence,
    ) -> (AmbientVector, AmbientVector, AmbientVector, f64, bool) {
        let raw_n1 = self.n1(g);
        let weighted_n1 = raw_n1 * valence.deficit();
        let grief_load = weighted_n1.norm();
        let under_floor = grief_load < adaptive_purity_floor(valence);
        (raw_n1, weighted_n1, self.n2(&weighted_n1), grief_load, under_floor)
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
                for i in 0..AMBIENT_DIM { e[(i, k)] -= proj * e[(i, j)]; }
            }
            let norm = e.column(k).norm();
            if norm > MERCY_PURITY_FLOOR {
                for i in 0..AMBIENT_DIM { e[(i, k)] /= norm; }
            }
        }
        let gram: MercyGram = e.transpose() * &e;
        (e, (gram - MercyGram::identity()).norm())
    }
    pub fn purify(basis: &mut LivingMercyBasis) -> f64 {
        let (new_e, rho) = Self::reorthonormalize(&basis.e);
        basis.e = new_e; rho
    }
}
