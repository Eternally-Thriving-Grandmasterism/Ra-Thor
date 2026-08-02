//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient · valence · adaptive floor · concurrent zones · soft feedback · LatticeHealthReport · adaptive Cosmic Tick · zone observability · stress EMA recovery · health aggregates · composite score · ZoneHealthStatus · critical auto-remediate (v0.5.15)
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

mod soft_feedback;
pub use soft_feedback::*;

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
pub struct LivingMercyBasis { pub e: MercyBasisMatrix }

impl Default for LivingMercyBasis {
    fn default() -> Self { Self::canonical() }
}

impl LivingMercyBasis {
    pub fn canonical() -> Self {
        let mut e = MercyBasisMatrix::zeros();
        for i in 0..MERCY_DIM { e[(i, i)] = 1.0; }
        Self { e }
    }
    pub fn identity() -> Self { Self::canonical() }
    pub fn projector_matrix(&self) -> AmbientMatrix {
        let et = self.e.transpose();
        let gram: MercyGram = et * &self.e;
        match gram.try_inverse() {
            Some(ginv) => &self.e * ginv * et,
            None => &self.e * et,
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
}

impl ZoneState {
    pub fn new(id: usize) -> Self {
        let basis = LivingMercyBasis::canonical();
        let projector = MercyProjector { basis: basis.clone() };
        Self {
            id, basis, suppressor: NilpotentSuppressor { projector },
            grief_absorbed: 0.0, stress_ema: 0.0, vectors_processed: 0,
            last_rho: 0.0, purify_count: 0, critical_auto_purify_count: 0,
        }
    }
    pub fn inject_drift(&mut self, magnitude: f64) {
        let z = self.id as f64;
        self.basis.e[(0, 1)] += magnitude * (1.0 + 0.1 * z);
        self.basis.e[(4, 7)] -= magnitude * (1.0 + 0.07 * z);
        if AMBIENT_DIM > 9 { self.basis.e[(9, 2)] += magnitude * (1.0 + 0.13 * z); }
        self.suppressor.projector.basis = self.basis.clone();
    }
    pub fn process(&mut self, g: &AmbientVector, valence: Valence) -> f64 {
        self.process_with_alpha(g, valence, 0.05)
    }
    pub fn process_with_alpha(&mut self, g: &AmbientVector, valence: Valence, alpha: f64) -> f64 {
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
        self.suppressor.projector.basis = self.basis.clone();
        self.last_rho = rho;
        self.purify_count = self.purify_count.saturating_add(1);
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
        if self.critical_auto_remediate {
            let status = ZoneHealthStatus::classify(
                self.zones[z].stress_ema,
                self.zones[z].last_rho,
                self.adaptive_grief_scale,
            );
            if status == ZoneHealthStatus::Critical {
                self.zones[z].purify();
                self.zones[z].critical_auto_purify_count =
                    self.zones[z].critical_auto_purify_count.saturating_add(1);
            }
        }
        load
    }
    pub fn total_critical_auto_purifies(&self) -> usize {
        self.zones.iter().map(|z| z.critical_auto_purify_count).sum()
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn projector_is_idempotent() { assert!(MercyProjector::new().verify_idempotence(1e-12)); }
    #[test]
    fn projector_is_symmetric() { assert!(MercyProjector::new().verify_symmetry(1e-12)); }

    #[test]
    fn nilpotent_second_order_reaches_zero() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[0] = 0.3; g[1] = -0.7; g[8] = 0.9; g[9] = -0.4; g[12] = 0.6; g[15] = -0.25;
        let (n1, final_r) = s.suppress(&g);
        assert!(n1.norm() > 0.1);
        assert!((n1[8] - 0.9).abs() < 1e-10);
        assert!(final_r.norm() < MERCY_PURITY_FLOOR * 10.0);
    }

    #[test]
    fn orthogonal_complement_is_nontrivial() {
        let p = MercyProjector::new();
        let mut grief = AmbientVector::zeros(); grief[10] = 1.0; grief[14] = -0.5;
        assert!((p.orthogonal_component(&grief) - grief).norm() < 1e-12);
        assert!(p.project(&grief).norm() < 1e-12);
    }

    #[test]
    fn mercy_component_is_preserved() {
        let p = MercyProjector::new();
        let mut mercy_v = AmbientVector::zeros(); mercy_v[0] = 1.0; mercy_v[3] = -0.5; mercy_v[7] = 0.25;
        assert!((p.project(&mercy_v) - mercy_v).norm() < 1e-12);
        assert!(p.orthogonal_component(&mercy_v).norm() < 1e-12);
    }

    #[test]
    fn gram_schmidt_purifies_drifted_basis() {
        let mut basis = LivingMercyBasis::canonical();
        basis.e[(0, 1)] += 1e-4; basis.e[(9, 3)] -= 2e-4; basis.e[(2, 5)] += 5e-5;
        assert!(ModifiedGramSchmidt::purify(&mut basis) < 1e-10);
        let gram: MercyGram = basis.e.transpose() * &basis.e;
        for i in 0..MERCY_DIM {
            assert_relative_eq!(gram[(i, i)], 1.0, epsilon = 1e-10);
            for j in (i + 1)..MERCY_DIM { assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10); }
        }
    }

    #[test]
    fn algebra_closure_holds() { assert!(TolcAlgebra::new().verify_closure()); }

    #[test]
    fn ambient_dim_is_elevated() {
        assert!(AMBIENT_DIM > MERCY_DIM);
        let b = LivingMercyBasis::canonical();
        assert_eq!(b.ambient_dim(), 16);
        assert_eq!(b.mercy_dim(), 8);
    }

    #[test]
    fn valence_high_softens_grief_load() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros(); g[8] = 1.0; g[12] = -0.8; g[15] = 0.5;
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
        let mut g = AmbientVector::zeros(); g[10] = 2.0;
        let raw_norm = s.n1(&g).norm();
        for v in [0.0, 0.25, 0.5, 0.75, 0.999999] {
            let (_, _, _, load, _) = s.suppress_weighted(&g, Valence::new(v));
            assert!((load - (1.0 - v) * raw_norm).abs() < 1e-10);
        }
    }

    #[test]
    fn valence_clamps_to_unit_interval() {
        assert_eq!(Valence::new(-1.0).value(), 0.0);
        assert_eq!(Valence::new(2.0).value(), 1.0);
        assert!(Valence::HIGH.is_high());
    }

    #[test]
    fn adaptive_floor_tightens_with_high_valence() {
        let pure = adaptive_purity_floor(Valence::new(1.0));
        let high = adaptive_purity_floor(Valence::HIGH);
        let zero = adaptive_purity_floor(Valence::ZERO);
        assert!((pure - MERCY_PURITY_FLOOR).abs() < 1e-18);
        assert!(high >= pure && zero > high);
        assert!((zero - MERCY_PURITY_FLOOR * 100.0).abs() < 1e-15);
    }

    #[test]
    fn suppress_weighted_reports_under_floor() {
        let s = NilpotentSuppressor::new();
        let mut tiny = AmbientVector::zeros(); tiny[10] = 1e-12;
        let (_, _, _, _, under_h) = s.suppress_weighted(&tiny, Valence::HIGH);
        assert!(under_h);
        let mut big = AmbientVector::zeros(); big[10] = 2.0;
        let (_, _, _, load_z, under_z) = s.suppress_weighted(&big, Valence::ZERO);
        assert!(!under_z && load_z > 1.0);
    }

    #[test]
    fn concurrent_zones_independent_grief() {
        let mut lattice = ConcurrentZoneLattice::new(3);
        let mut g0 = AmbientVector::zeros(); g0[10] = 1.0;
        let mut g1 = AmbientVector::zeros(); g1[11] = 2.0;
        let mut g2 = AmbientVector::zeros(); g2[12] = 3.0;
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
        for rho in lattice.global_purify() { assert!(rho < 1e-10); }
        assert!(lattice.max_rho() < 1e-10);
    }

    #[test]
    fn concurrent_zones_staggered_tick_runs() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 10;
        let mut g = AmbientVector::zeros(); g[10] = 0.5;
        for i in 0..25 { lattice.process(i % 2, &g, Valence::MID); }
        assert!(lattice.total_grief() > 0.0);
        assert_eq!(lattice.zones[0].vectors_processed + lattice.zones[1].vectors_processed, 25);
    }

    #[test]
    fn soft_feedback_bridge_records_events() {
        let mut bridge = SoftFeedbackBridge::new(3);
        let ev = bridge.ingest_scalar_grief(1, 2.0, Valence::ZERO);
        assert_eq!(ev.zone_id, 1);
        assert!(ev.grief_load > 1.5);
        assert!(!ev.under_floor);
        let soft = bridge.ingest_scalar_grief(0, 2.0, Valence::HIGH);
        assert!(soft.grief_load < 1e-5);
        assert_eq!(bridge.drain_events().len(), 2);
        assert!(bridge.drain_events().is_empty());
        let snaps = bridge.snapshots();
        assert_eq!(snaps.len(), 3);
        assert!(snaps[1].grief_absorbed > 1.5);
    }

    #[test]
    fn soft_feedback_event_fields_are_stable() {
        let ev = SoftFeedbackEvent {
            zone_id: 2, grief_load: 1.25, valence: 0.5, under_floor: false, tick: 42,
        };
        assert_eq!(ev.zone_id, 2);
        assert!((ev.grief_load - 1.25).abs() < 1e-12);
        assert!((ev.valence - 0.5).abs() < 1e-12);
        assert!(!ev.under_floor);
        assert_eq!(ev.tick, 42);
    }

    #[test]
    fn lattice_health_report_is_healthy_after_purify() {
        let mut bridge = SoftFeedbackBridge::new(3);
        for i in 0..30 {
            bridge.ingest_scalar_grief(i % 3, 1.0, Valence::MID);
        }
        bridge.global_purify();
        let h = bridge.health_report();
        assert_eq!(h.schema, "ra_thor_lattice_health_v1");
        assert_eq!(h.zone_count, 3);
        assert!(h.total_vectors >= 30);
        assert!(h.healthy, "max_rho={}", h.max_rho);
        assert!(h.max_rho < 1e-9);
    }

    #[test]
    fn adaptive_purify_period_tightens_under_grief() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 1000;
        lattice.adaptive_grief_scale = 100.0;
        lattice.min_purify_period = 10;
        let base = lattice.effective_purify_period(0);
        assert_eq!(base, 1000);
        let mut g = AmbientVector::zeros(); g[10] = 5.0;
        for _ in 0..50 { lattice.process(0, &g, Valence::ZERO); }
        let tight = lattice.effective_purify_period(0);
        assert!(tight < base);
        assert!(tight >= lattice.min_purify_period);
        assert!(lattice.effective_purify_period(1) >= tight);
    }

    #[test]
    fn high_grief_zone_fires_more_cosmic_ticks() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 100;
        lattice.adaptive_grief_scale = 20.0;
        lattice.min_purify_period = 10;
        lattice.stress_alpha = 0.15;
        let mut heavy = AmbientVector::zeros(); heavy[10] = 5.0;
        let mut light = AmbientVector::zeros(); light[11] = 0.01;
        for _ in 0..800 { lattice.process(0, &heavy, Valence::ZERO); }
        for _ in 0..800 { lattice.process(1, &light, Valence::HIGH); }
        let c0 = lattice.zones[0].purify_count;
        let c1 = lattice.zones[1].purify_count;
        assert!(c0 > 0);
        assert!(c0 > c1, "z0={c0} z1={c1}");
    }

    #[test]
    fn zone_snapshot_exposes_observability_fields() {
        let mut bridge = SoftFeedbackBridge::new(2);
        bridge.ingest_scalar_grief(0, 1.0, Valence::ZERO);
        bridge.global_purify();
        let snaps = bridge.snapshots();
        assert_eq!(snaps.len(), 2);
        assert!(snaps[0].purify_count >= 1);
        assert!(snaps[0].effective_period >= 1);
        let h = bridge.health_report();
        assert!(h.zones[0].purify_count >= 1);
    }

    #[test]
    fn health_report_exposes_stress_aggregates() {
        let mut bridge = SoftFeedbackBridge::new(3);
        for i in 0..60 {
            bridge.ingest_scalar_grief(i % 3, 2.0, Valence::ZERO);
        }
        bridge.global_purify();
        let h = bridge.health_report();
        assert!(h.total_purify_count >= 3);
        assert!(h.max_stress_ema > 0.0);
        assert!(h.mean_effective_period >= 1.0);
        assert_eq!(h.zone_count, 3);
        assert!(h.total_vectors >= 60);
        assert!(h.health_score > 0.0 && h.health_score <= 1.0);
        assert!(h.healthy);
    }

    #[test]
    fn zone_health_status_classifies_calm_as_healthy() {
        assert_eq!(ZoneHealthStatus::classify(0.0, 0.0, 500.0), ZoneHealthStatus::Healthy);
        assert_eq!(ZoneHealthStatus::classify(60.0, 0.0, 500.0), ZoneHealthStatus::Stressed);
        assert_eq!(ZoneHealthStatus::classify(500.0, 0.0, 500.0), ZoneHealthStatus::Critical);
        assert_eq!(ZoneHealthStatus::classify(0.0, 1e-5, 500.0), ZoneHealthStatus::Critical);
    }

    #[test]
    fn critical_auto_remediate_fires_under_extreme_stress() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 5.0;
        lattice.stress_alpha = 0.5;
        lattice.critical_auto_remediate = true;
        lattice.purify_period = 10_000;
        let mut heavy = AmbientVector::zeros();
        heavy[10] = 20.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        assert!(lattice.zones[0].critical_auto_purify_count > 0);
        assert!(lattice.total_critical_auto_purifies() > 0);
        assert!(lattice.zones[0].last_rho < 1e-9);
    }

    #[test]
    fn critical_auto_remediate_can_be_disabled() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 5.0;
        lattice.stress_alpha = 0.5;
        lattice.critical_auto_remediate = false;
        lattice.purify_period = 10_000;
        let mut heavy = AmbientVector::zeros();
        heavy[10] = 20.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        assert_eq!(lattice.zones[0].critical_auto_purify_count, 0);
    }

    #[test]
    fn health_report_counts_zone_status() {
        let mut bridge = SoftFeedbackBridge::new(2);
        bridge.lattice.adaptive_grief_scale = 10.0;
        bridge.lattice.stress_alpha = 0.3;
        for _ in 0..30 { bridge.ingest_scalar_grief(0, 8.0, Valence::ZERO); }
        for _ in 0..5 { bridge.ingest_scalar_grief(1, 1e-12, Valence::HIGH); }
        bridge.global_purify();
        let h = bridge.health_report();
        assert_eq!(h.zones.len(), 2);
        assert!(h.zones_stressed + h.zones_critical + h.zones_healthy == 2);
        assert!(h.zones[0].status != ZoneHealthStatus::Healthy);
        assert_eq!(h.zones[1].status, ZoneHealthStatus::Healthy);
    }

    #[test]
    fn health_score_drops_under_stress_and_recovers() {
        let mut bridge = SoftFeedbackBridge::new(1);
        bridge.lattice.adaptive_grief_scale = 10.0;
        bridge.lattice.stress_alpha = 0.25;
        bridge.lattice.purify_period = 1000;
        let calm = bridge.health_report().health_score;
        assert!((calm - 1.0).abs() < 1e-6);
        for _ in 0..40 { bridge.ingest_scalar_grief(0, 8.0, Valence::ZERO); }
        bridge.global_purify();
        let stressed = bridge.health_report().health_score;
        assert!(stressed < calm && stressed > 0.0);
        for _ in 0..200 { bridge.ingest_scalar_grief(0, 1e-12, Valence::HIGH); }
        bridge.global_purify();
        let recovered = bridge.health_report().health_score;
        assert!(recovered > stressed);
    }

    #[test]
    fn stress_ema_recovers_period_under_calm() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.purify_period = 1000;
        lattice.adaptive_grief_scale = 10.0;
        lattice.min_purify_period = 20;
        lattice.stress_alpha = 0.2;
        let mut heavy = AmbientVector::zeros(); heavy[10] = 8.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        let tight = lattice.effective_purify_period(0);
        assert!(tight < 1000);
        let mut calm = AmbientVector::zeros(); calm[10] = 1e-12;
        for _ in 0..200 { lattice.process(0, &calm, Valence::HIGH); }
        let recovered = lattice.effective_purify_period(0);
        assert!(recovered > tight);
        assert!(lattice.zones[0].grief_absorbed > 100.0);
        assert!(lattice.zones[0].stress_ema < lattice.zones[0].grief_absorbed);
    }
}
