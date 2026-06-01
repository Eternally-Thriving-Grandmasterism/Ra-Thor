//! adapter.rs
//!
//! State persistence with forward migration + rollback support.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    version: u32,
    name: &'static str,
    current_valence: Valence,
    #[serde(default)]
    blessings_count: u32,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_count: 0,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_count
    }

    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(data) = fs::read_to_string(path) {
            // Try current version
            if let Ok(mut adapter) = serde_json::from_str::<ShardComposerAdapter>(&data) {
                if adapter.version == CURRENT_VERSION {
                    return adapter;
                }

                if adapter.version < CURRENT_VERSION {
                    // Forward migration
                    return Self::migrate_forward(adapter);
                }

                if adapter.version > CURRENT_VERSION {
                    // Rollback / downgrade scenario
                    println!("[Warning] Saved state is from a newer version (v{}). Attempting best-effort rollback.", adapter.version);
                    return Self::attempt_rollback(adapter);
                }
            }

            // Last resort: try raw JSON migration
            if let Ok(old) = serde_json::from_str::<serde_json::Value>(&data) {
                return Self::migrate_from_json(old);
            }
        }
        Self::new()
    }

    fn migrate_forward(old: ShardComposerAdapter) -> Self {
        // Simple forward migration (expand as needed)
        let mut new = Self::new();
        new.blessings_count = old.blessings_count;
        new.current_valence = old.current_valence;
        new
    }

    fn attempt_rollback(newer: ShardComposerAdapter) -> Self {
        // Best-effort rollback: keep what we can
        let mut adapter = Self::new();
        adapter.blessings_count = newer.blessings_count;
        adapter.current_valence = newer.current_valence;

        // Note: Any fields only present in newer versions will be lost.
        println!("[Rollback] Loaded with potential data loss from newer format.");
        adapter
    }

    fn migrate_from_json(old: serde_json::Value) -> Self {
        let mut adapter = Self::new();
        if let Some(count) = old.get("blessings_count").or_else(|| old.get("blessings_received")).and_then(|v| v.as_u64()) {
            adapter.blessings_count = count as u32;
        }
        adapter
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
