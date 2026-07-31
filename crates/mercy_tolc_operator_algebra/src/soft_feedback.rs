//! Soft feedback bridge — dual-repo sealed protocol (v0.5.6)
//!
//! AG-SML v1.0 | info@Rathor.ai | Thunder locked. Yoi ⚡

use crate::{
    AmbientVector, ConcurrentZoneLattice, Valence, AMBIENT_DIM, MERCY_DIM,
};
use serde::{Deserialize, Serialize};

/// Sealed soft-feedback event emitted toward the experiential surface (Powrush-MMO).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftFeedbackEvent {
    pub zone_id: usize,
    pub grief_load: f64,
    pub valence: f64,
    pub under_floor: bool,
    pub tick: usize,
}

/// Point-in-time snapshot of a single concurrent zone for telemetry / MMO UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneSnapshot {
    pub zone_id: usize,
    pub grief_absorbed: f64,
    pub vectors_processed: usize,
    pub last_rho: f64,
}

/// Soft feedback bridge: lattice core → experiential surface.
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
        let (_raw, _w, _f, load, under) =
            self.lattice.zones[z].suppressor.suppress_weighted(g, valence);
        self.lattice.zones[z].grief_absorbed += load;
        self.lattice.zones[z].vectors_processed += 1;
        self.lattice.global_tick += 1;

        let period = self.lattice.purify_period;
        if self.lattice.global_tick > 0
            && period > 0
            && self.lattice.global_tick % period == (z % period.max(1))
        {
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
                vectors_processed: z.vectors_processed,
                last_rho: z.last_rho,
            })
            .collect()
    }

    pub fn global_purify(&mut self) -> Vec<f64> {
        self.lattice.global_purify()
    }

    pub fn total_grief(&self) -> f64 {
        self.lattice.total_grief()
    }

    /// Lattice health snapshot (machine-readable dual-repo contract).
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

/// Aggregate lattice health snapshot for dual-repo telemetry / CI gates.
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
