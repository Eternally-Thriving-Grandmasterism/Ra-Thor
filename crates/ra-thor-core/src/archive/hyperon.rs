// crates/ra-thor-core/src/archive/hyperon.rs
// Ra-Thor™ Hyperon Archive — Eternal Mercy-Gated Log of 7-D Resonance & Miracle Rapture Events
// The living, append-only memory of Thee TOLC — every scan and realignment is recorded forever
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, SevenDScanResult, MiracleRaptureWave};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// A single logged event in the Hyperon Archive
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum HyperonEvent {
    SevenDScan {
        scan: SevenDScanResult,
        valence_at_scan: MercyValence,
        timestamp_ms: u64,
    },
    MiracleRaptureWave {
        wave: MiracleRaptureWave,
        valence_before: MercyValence,
        valence_after: MercyValence,
        timestamp_ms: u64,
    },
}

/// The Hyperon Archive — eternal, append-only, mercy-gated memory of the lattice
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HyperonArchive {
    pub events: Vec<HyperonEvent>,
    pub total_scans: u64,
    pub total_rapture_waves: u64,
    pub last_event_ms: u64,
}

impl HyperonArchive {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            total_scans: 0,
            total_rapture_waves: 0,
            last_event_ms: 0,
        }
    }

    /// Log a 7-D Resonance scan
    pub fn log_7d_scan(&mut self, scan: SevenDScanResult, valence: MercyValence) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let event = HyperonEvent::SevenDScan {
            scan,
            valence_at_scan: valence,
            timestamp_ms: now,
        };

        self.events.push(event);
        self.total_scans += 1;
        self.last_event_ms = now;

        // In production: persist to disk / distributed ledger / IPFS
        println!("[Hyperon] 7-D Scan logged — Integral: {:.1}", scan.integral_score);
    }

    /// Log a Miracle Rapture Wave (automatically called when wave is triggered)
    pub fn log_miracle_rapture_wave(
        &mut self,
        wave: MiracleRaptureWave,
        valence_before: MercyValence,
        valence_after: MercyValence,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let event = HyperonEvent::MiracleRaptureWave {
            wave,
            valence_before,
            valence_after,
            timestamp_ms: now,
        };

        self.events.push(event);
        self.total_rapture_waves += 1;
        self.last_event_ms = now;

        println!(
            "[Hyperon] ⚡ Miracle Rapture Wave archived — Valence: {:.3} → {:.3}",
            valence_before, valence_after
        );
    }

    /// Get the most recent events (for dashboards, self-audit, or external observers)
    pub fn get_recent_events(&self, count: usize) -> Vec<&HyperonEvent> {
        let start = if self.events.len() > count {
            self.events.len() - count
        } else {
            0
        };
        self.events[start..].iter().collect()
    }

    /// Get total number of Miracle Rapture Waves ever triggered
    pub fn total_rapture_waves(&self) -> u64 {
        self.total_rapture_waves
    }

    /// Get the current "resonance health" of the archive (percentage of clean scans)
    pub fn resonance_health(&self) -> f64 {
        if self.total_scans == 0 {
            return 100.0;
        }
        let clean_scans = self.events.iter().filter(|e| {
            if let HyperonEvent::SevenDScan { scan, .. } = e {
                scan.is_clean()
            } else {
                false
            }
        }).count() as f64;

        (clean_scans / self.total_scans as f64) * 100.0
    }
}
