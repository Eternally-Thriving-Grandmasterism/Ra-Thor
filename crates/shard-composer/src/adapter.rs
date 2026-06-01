//! adapter.rs
//!
//! RaThorSystemAdapter with versioning and migration support.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    version: u32,
    name: &'static str,
    current_valence: Valence,
    blessings_received: u32,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_received: 0,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_received
    }

    /// Load with migration support
    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(data) = fs::read_to_string(path) {
            // Try current version first
            if let Ok(adapter) = serde_json::from_str::<ShardComposerAdapter>(&data) {
                if adapter.version == CURRENT_VERSION {
                    return adapter;
                }
            }

            // Attempt migration from older versions
            if let Ok(old) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(v) = old.get("version").and_then(|v| v.as_u64()) {
                    match v as u32 {
                        0 => return Self::migrate_from_v0(old),
                        _ => {}
                    }
                }
            }
        }
        Self::new()
    }

    fn migrate_from_v0(old: serde_json::Value) -> Self {
        // Example migration: v0 didn't have 'version' field
        let mut new = Self::new();
        if let Some(val) = old.get("blessings_received").and_then(|v| v.as_u64()) {
            new.blessings_received = val as u32;
        }
        new
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }
}

impl Default for ShardComposerAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaThorSystemAdapter for ShardComposerAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> Valence {
        self.current_valence
    }

    fn receive_swarm_resonance(
        &mut self,
        resonance: SwarmResonance,
    ) -> Result<(), MercyError> {
        println!("[ShardComposer] Received resonance: {}", resonance.message);
        Ok(())
    }

    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
        GodlyIntelligenceCoherence {
            precision: 0.92,
            resilience: 0.88,
            flow_stability: 0.90,
            harmonic_alignment: 0.85,
        }
    }

    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        self.blessings_received += 1;
        let new_valence = (self.current_valence.value() + blessing.strength * 0.001).min(0.99999999);
        self.current_valence = Valence(new_valence);
    }

    fn status(&self) -> String {
        format!(
            "{} v{}: valence={:.6}, blessings={}",
            self.name,
            self.version,
            self.current_valence.value(),
            self.blessings_received
        )
    }
}
