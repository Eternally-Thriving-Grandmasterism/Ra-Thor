//! Soft feedback bridge — dual-repo sealed protocol (v0.5.4)
//!
//! AG-SML v1.0 | info@Rathor.ai | Thunder locked. Yoi ⚡

use crate::{
    AmbientVector, ConcurrentZoneLattice, Valence, AMBIENT_DIM, MERCY_DIM,
};
use serde::{Deserialize, Serialize};

/// Sealed soft-feedback event emitted toward the experiential surface (Powrush-MMO).
///
/// Serde shape is the dual-repo contract. Powrush simulation mode mirrors these
/// fields without a hard crate dependency on Ra-Thor.
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
///
/// Holds a [`ConcurrentZoneLattice`], records sealed [`SoftFeedbackEvent`]s, and
/// exposes drain/snapshot APIs for the Powrush-MMO soft feedback loop.
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

    /// Ingest one ambient grief vector under valence into `zone_id`.
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

    /// Convenience: ingest pure orthogonal grief energy on a single ambient coord.
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

    /// Drain all pending events (clears the buffer).
    pub fn drain_events(&mut self) -> Vec<SoftFeedbackEvent> {
        std::mem::take(&mut self.events)
    }

    /// Snapshot every zone for telemetry / dual-repo UI.
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
}
