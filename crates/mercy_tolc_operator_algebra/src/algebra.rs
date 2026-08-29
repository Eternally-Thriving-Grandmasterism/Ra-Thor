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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneState {
    pub id: usize,
    pub basis: LivingMercyBasis,
    pub suppressor: NilpotentSuppressor,
    pub grief_absorbed: f64,
    pub stress_ema: f64,
    pub vectors_processed: usize,
    pub last_rho: f64,
    pub purify_count: usize,
    pub critical_auto_purify_count: usize,
    pub soft_remediate_count: usize,
}

impl ZoneState {
    pub fn new(id: usize) -> Self {
        let basis = LivingMercyBasis::canonical();
        let projector = MercyProjector { basis: basis.clone() };
        Self {
            id, basis, suppressor: NilpotentSuppressor { projector },
            grief_absorbed: 0.0, stress_ema: 0.0, vectors_processed: 0,
            last_rho: 0.0, purify_count: 0, critical_auto_purify_count: 0,
            soft_remediate_count: 0,
        }
    }
    pub fn inject_drift(&mut self, magnitude: f64) {
        let z = self.id as f64;
        self.basis.e[(0, 1)] += magnitude * (1.0 + 0.1 * z);
        self.basis.e[(4, 7)] -= magnitude * (1.0 + 0.07 * z);
        if AMBIENT_DIM > 9 { self.basis.e[(9, 2)] += magnitude * (1.0 + 0.13 * z); }
        self.last_rho = self.basis.gram_residual();
        self.basis.reschedule_tikhonov(self.stress_ema);
        self.suppressor.projector.basis = self.basis.clone();
    }
    pub fn process(&mut self, g: &AmbientVector, valence: Valence) -> f64 {
        self.process_with_alpha(g, valence, 0.05)
    }
    pub fn process_with_alpha(&mut self, g: &AmbientVector, valence: Valence, alpha: f64) -> f64 {
        self.basis.reschedule_tikhonov(self.stress_ema);
        self.suppressor.projector.basis = self.basis.clone();
        let (_r, _w, _f, load, _) = self.suppressor.suppress_weighted(g, valence);
        self.grief_absorbed += load;
        let a = alpha.clamp(0.0, 1.0);
        self.stress_ema = (1.0 - a) * self.stress_ema + a * load;
        self.vectors_processed += 1;
        load
    }
    pub fn decay_stress(&mut self, alpha: f64) {
        let a = alpha.clamp(0.0, 1.0);
        self.stress_ema *= 1.0 - a;
    }
    pub fn purify(&mut self) -> f64 {
        let rho = ModifiedGramSchmidt::purify(&mut self.basis);
        self.last_rho = rho;
        self.purify_count = self.purify_count.saturating_add(1);
        self.basis.tikhonov_lambda = schedule_tikhonov_lambda(rho, 0.0);
        self.suppressor.projector.basis = self.basis.clone();
        rho
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConcurrentZoneLattice {
    pub zones: Vec<ZoneState>,
    pub global_tick: usize,
    pub purify_period: usize,
    pub adaptive_grief_scale: f64,
    pub min_purify_period: usize,
    pub stress_alpha: f64,
    pub critical_auto_remediate: bool,
    pub soft_remediate_stressed: bool,
    pub soft_remediate_alpha: f64,
}

impl ConcurrentZoneLattice {
    pub fn new(n_zones: usize) -> Self {
        let n = n_zones.max(1);
        let mut zones: Vec<ZoneState> = (0..n).map(ZoneState::new).collect();
        for z in zones.iter_mut() { z.inject_drift(3e-5); }
        Self {
            zones, global_tick: 0, purify_period: 2_500,
            adaptive_grief_scale: 500.0, min_purify_period: 50, stress_alpha: 0.05,
            critical_auto_remediate: true,
            soft_remediate_stressed: true,
            soft_remediate_alpha: 0.15,
        }
    }
    pub fn zone_count(&self) -> usize { self.zones.len() }
    pub fn effective_purify_period(&self, zone_id: usize) -> usize {
        let z = zone_id % self.zones.len().max(1);
        let stress = self.zones[z].stress_ema;
        let scale = self.adaptive_grief_scale.max(1e-9);
        let factor = 1.0 + stress / scale;
        let adaptive = (self.purify_period as f64 / factor).round() as usize;
        adaptive.max(self.min_purify_period).max(1)
    }
    pub fn process(&mut self, zone_id: usize, g: &AmbientVector, valence: Valence) -> f64 {
        let z = zone_id % self.zones.len();
        let alpha = self.stress_alpha;
        let load = self.zones[z].process_with_alpha(g, valence, alpha);
        self.global_tick += 1;
        let decay = alpha * 0.25;
        for (i, zone) in self.zones.iter_mut().enumerate() {
            if i != z { zone.decay_stress(decay); }
        }
        let period = self.effective_purify_period(z);
        if self.global_tick > 0 && self.global_tick % period == (z % period) {
            self.zones[z].purify();
        }
        let status = ZoneHealthStatus::classify(
            self.zones[z].stress_ema,
            self.zones[z].last_rho,
            self.adaptive_grief_scale,
        );
        if status == ZoneHealthStatus::Critical && self.critical_auto_remediate {
            self.zones[z].purify();
            self.zones[z].critical_auto_purify_count =
                self.zones[z].critical_auto_purify_count.saturating_add(1);
        } else if status == ZoneHealthStatus::Stressed && self.soft_remediate_stressed {
            let a = self.soft_remediate_alpha.clamp(0.0, 1.0);
            self.zones[z].decay_stress(a);
            self.zones[z].soft_remediate_count =
                self.zones[z].soft_remediate_count.saturating_add(1);
        }
        load
    }
    pub fn total_critical_auto_purifies(&self) -> usize {
        self.zones.iter().map(|z| z.critical_auto_purify_count).sum()
    }
    pub fn total_soft_remediates(&self) -> usize {
        self.zones.iter().map(|z| z.soft_remediate_count).sum()
    }
    pub fn global_purify(&mut self) -> Vec<f64> {
        self.zones.iter_mut().map(|z| z.purify()).collect()
    }
    pub fn max_rho(&self) -> f64 {
        self.zones.iter().map(|z| z.last_rho).fold(0.0_f64, f64::max)
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
pub struct ValenceOperator { pub k: usize }

#[derive(Clone, Debug)]
pub struct TolcProjector(pub MercyProjector);

impl TolcProjector {
    pub fn project_consciousness(&self, operator: &ValenceOperator) -> ValenceOperator { operator.clone() }
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
            suppressor: NilpotentSuppressor { projector: mercy.clone() },
            mercy,
        }
    }
    pub fn create_positive_valence(&self, k: usize) -> ValenceOperator {
        self.tolc.project_consciousness(&ValenceOperator { k })
    }
    pub fn swarm_consensus(&self, n_probes: usize) -> f64 {
        let cv = (n_probes as f64).sqrt();
        if (cv - 1.0).abs() < MERCY_THRESHOLD { 1.0 }
        else { 1.0 - (cv - 1.0).signum() * (cv - 1.0).powi(2) * 0.01 }
    }
    pub fn verify_closure(&self) -> bool {
        self.mercy.verify_idempotence(1e-10) && self.mercy.verify_symmetry(1e-10)
    }
}

impl Default for TolcAlgebra {
    fn default() -> Self { Self::new() }
}
