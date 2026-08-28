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
