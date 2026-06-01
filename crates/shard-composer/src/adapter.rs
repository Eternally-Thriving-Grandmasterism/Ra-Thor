//! adapter.rs
//!
//! Advanced state migration including field renaming support.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 2; // Bumped for field rename example

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    version: u32,
    name: &'static str,
    current_valence: Valence,

    // New field name (v2+)
    #[serde(default)]
    blessings_count: u32,

    // Keep old name during transition for migration
    #[serde(rename = "blessings_received", alias = "blessings_received")]
    #[serde(skip_serializing)]
    _old_blessings: Option<u32>,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_count: 0,
            _old_blessings: None,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_count
    }

    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(adapter) = serde_json::from_str::<ShardComposerAdapter>(&data) {
                if adapter.version == CURRENT_VERSION {
                    return adapter;
                }
            }

            if let Ok(old) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(v) = old.get("version").and_then(|v| v.as_u64()) {
                    match v as u32 {
                        0 | 1 => return Self::migrate_from_v1(old),
                        _ => {}
                    }
                }
            }
        }
        Self::new()
    }

    fn migrate_from_v1(old: serde_json::Value) -> Self {
        let mut new = Self::new();

        // Handle renamed field: blessings_received -> blessings_count
        if let Some(count) = old.get("blessings_received").and_then(|v| v.as_u64()) {
            new.blessings_count = count as u32;
        } else if let Some(count) = old.get("blessings_count").and_then(|v| v.as_u64()) {
            new.blessings_count = count as u32;
        }

        if let Some(val) = old.get("current_valence").and_then(|v| v.as_f64()) {
            new.current_valence = Valence(val);
        }

        new
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
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
        self.blessings_count += 1;
        let new_valence = (self.current_valence.value() + blessing.strength * 0.001).min(0.99999999);
        self.current_valence = Valence(new_valence);
    }

    fn status(&self) -> String {
        format!(
            "{} v{}: valence={:.6}, blessings={}",
            self.name,
            self.version,
            self.current_valence.value(),
            self.blessings_count
        )
    }
}
