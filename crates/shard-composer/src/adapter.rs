//! adapter.rs
//!
//! State persistence with versioning, migration, and SHA256 checksum verification.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    version: u32,
    checksum: String,
    data: serde_json::Value,
}

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

    fn compute_checksum(data: &serde_json::Value) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_value(self)?;
        let checksum = Self::compute_checksum(&data);

        let persisted = PersistedState {
            version: self.version,
            checksum,
            data,
        };

        let json = serde_json::to_string_pretty(&persisted)?;
        fs::write(path, json)
    }

    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(persisted) = serde_json::from_str::<PersistedState>(&content) {
                // Verify checksum
                let computed = Self::compute_checksum(&persisted.data);
                if computed != persisted.checksum {
                    eprintln!("[Security] State checksum mismatch! Possible corruption or tampering.");
                    return Self::new();
                }

                if let Ok(mut adapter) = serde_json::from_value::<ShardComposerAdapter>(persisted.data) {
                    if adapter.version == CURRENT_VERSION {
                        return adapter;
                    }
                    if adapter.version < CURRENT_VERSION {
                        return Self::migrate_forward(adapter);
                    }
                    if adapter.version > CURRENT_VERSION {
                        eprintln!("[Warning] Newer state version detected. Attempting rollback.");
                        return Self::attempt_rollback(adapter);
                    }
                }
            }
        }
        Self::new()
    }

    // ... (migrate_forward and attempt_rollback remain similar)
    fn migrate_forward(old: ShardComposerAdapter) -> Self {
        let mut new = Self::new();
        new.blessings_count = old.blessings_count;
        new.current_valence = old.current_valence;
        new
    }

    fn attempt_rollback(newer: ShardComposerAdapter) -> Self {
        let mut adapter = Self::new();
        adapter.blessings_count = newer.blessings_count;
        adapter.current_valence = newer.current_valence;
        adapter
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
