//! adapter.rs
//!
//! RaThorSystemAdapter implementation for shard-composer.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    name: &'static str,
    current_valence: Valence,
    blessings_received: u32,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_received: 0,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_received
    }

    /// Save adapter state to a file for persistence across xtask runs
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    /// Load adapter state from file, or create new if file doesn't exist
    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(adapter) = serde_json::from_str(&data) {
                return adapter;
            }
        }
        Self::new()
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
        println!(
            "[ShardComposer] Applied epigenetic blessing: {} (strength {:.2}) — Total: {}",
            blessing.blessing_type, blessing.strength, self.blessings_received
        );

        let new_valence = (self.current_valence.value() + blessing.strength * 0.001).min(0.99999999);
        self.current_valence = Valence(new_valence);
    }

    fn status(&self) -> String {
        format!(
            "{}: valence={:.6}, blessings={}",
            self.name,
            self.current_valence.value(),
            self.blessings_received
        )
    }
}
