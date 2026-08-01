//! Soft feedback bridge — dual-repo sealed protocol (v0.5.9)
//!
//! AG-SML v1.0 | info@Rathor.ai | Thunder locked. Yoi ⚡

use crate::{
    AmbientVector, ConcurrentZoneLattice, Valence, AMBIENT_DIM, MERCY_DIM,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftFeedbackEvent {
    pub zone_id: usize,
    pub grief_load: f64,
    pub valence: f64,
    pub under_floor: bool,
    pub tick: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneSnapshot {
    pub zone_id: usize,
    pub grief_absorbed: f64,
    pub stress_ema: f64,
    pub vectors_processed: usize,
    pub last_rho: f64,
    pub purify_count: usize,
    pub effective_period: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftFeedbackBridge {
    pub lattice: ConcurrentZoneLattice,
    pub events: Vec<SoftFeedbackEvent>,
    pub max_events: usize,
}

impl SoftFeedbackBridge {
    pub fn new(n_zones: usize) -> Self {
        Self {
            lattice: ConcurrentZoneLattice::new(n_zones),
            events: Vec::new(),
            max_events: 10_000,
        }
    }

    pub fn ingest(
        &mut self,
        zone_id: usize,
        g: &AmbientVector,
        valence: Valence,
    ) -> SoftFeedbackEvent {
        let z = zone_id % self.lattice.zone_count().max(1);
        let alpha = self.lattice.stress_alpha;
        let (_raw, _w, _f, load, under) =
            self.lattice.zones[z].suppressor.suppress_weighted(g, valence);
        self.lattice.zones[z].grief_absorbed += load;
        let a = alpha.clamp(0.0, 1.0);
        self.lattice.zones[z].stress_ema =
            (1.0 - a) * self.lattice.zones[z].stress_ema + a * load;
        self.lattice.zones[z].vectors_processed += 1;
        self.lattice.global_tick += 1;

        // Mild decay on sibling zones (calm recovery)
        let decay = a * 0.25;
        let n = self.lattice.zones.len();
        for i in 0..n {
            if i != z {
                self.lattice.zones[i].decay_stress(decay);
            }
        }

        let period = self.lattice.effective_purify_period(z);
        if self.lattice.global_tick > 0 && self.lattice.global_tick % period == (z % period) {
            self.lattice.zones[z].purify();
        }

        let ev = SoftFeedbackEvent {
            zone_id: z,
            grief_load: load,
            valence: valence.value(),
            under_floor: under,
            tick: self.lattice.global_tick,
        };
        self.events.push(ev.clone());
        if self.events.len() > self.max_events {
            let overflow = self.events.len() - self.max_events;
            self.events.drain(0..overflow);
        }
        ev
    }

    pub fn ingest_scalar_grief(
        &mut self,
        zone_id: usize,
        orthogonal_energy: f64,
        valence: Valence,
    ) -> SoftFeedbackEvent {
        let mut g = AmbientVector::zeros();
        let coord = MERCY_DIM + (zone_id % (AMBIENT_DIM - MERCY_DIM).max(1));
        g[coord] = orthogonal_energy;
        self.ingest(zone_id, &g, valence)
    }

    pub fn drain_events(&mut self) -> Vec<SoftFeedbackEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn snapshots(&self) -> Vec<ZoneSnapshot> {
        self.lattice
            .zones
            .iter()
            .map(|z| ZoneSnapshot {
                zone_id: z.id,
                grief_absorbed: z.grief_absorbed,
                stress_ema: z.stress_ema,
                vectors_processed: z.vectors_processed,
                last_rho: z.last_rho,
                purify_count: z.purify_count,
                effective_period: self.lattice.effective_purify_period(z.id),
            })
            .collect()
    }

    pub fn global_purify(&mut self) -> Vec<f64> {
        self.lattice.global_purify()
    }

    pub fn total_grief(&self) -> f64 {
        self.lattice.total_grief()
    }

    pub fn health_report(&self) -> LatticeHealthReport {
        let zones = self.snapshots();
        let total_vectors: usize = zones.iter().map(|z| z.vectors_processed).sum();
        let max_rho = self.lattice.max_rho();
        LatticeHealthReport {
            schema: "ra_thor_lattice_health_v1".to_string(),
            ambient_dim: AMBIENT_DIM,
            mercy_dim: MERCY_DIM,
            zone_count: self.lattice.zone_count(),
            global_tick: self.lattice.global_tick,
            total_grief: self.total_grief(),
            total_vectors,
            max_rho,
            pending_events: self.events.len(),
            zones,
            healthy: max_rho < 1e-9,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeHealthReport {
    pub schema: String,
    pub ambient_dim: usize,
    pub mercy_dim: usize,
    pub zone_count: usize,
    pub global_tick: usize,
    pub total_grief: f64,
    pub total_vectors: usize,
    pub max_rho: f64,
    pub pending_events: usize,
    pub zones: Vec<ZoneSnapshot>,
    pub healthy: bool,
}
