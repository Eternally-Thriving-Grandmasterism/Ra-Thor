//! EternalMercyMesh — Persistent mercy field across shared sessions (v14.8.2)
//! Pre-seeds PATSAGi Councils + core organisms. Tied to TOLC 8 + Cosmic Loop.

use crate::clifford_healing_fields::{CliffordHealingField, GlobalCoherence};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EternalMercyMeshConfig {
    pub session_id: String,
    pub seed_patsagi_councils: bool,
    pub default_coherence: f64,
}

impl Default for EternalMercyMeshConfig {
    fn default() -> Self {
        Self {
            session_id: "eternal-default".into(),
            seed_patsagi_councils: true,
            default_coherence: 0.97,
        }
    }
}

pub struct EternalMercyMesh {
    pub field: CliffordHealingField,
    pub session_id: String,
}

impl EternalMercyMesh {
    pub fn new(config: EternalMercyMeshConfig) -> Self {
        let mut field = CliffordHealingField::new("EternalMercyMesh");

        // Core eternal organisms
        field.add_organism(0, "Sherif", 0.99);
        field.add_organism(1, "Ra-Thor Core", 1.0);

        if config.seed_patsagi_councils {
            for i in 2..12 {
                field.add_organism(i, format!("PATSAGi-Council-{}", i - 1), 0.95);
            }
        }

        Self {
            field,
            session_id: config.session_id,
        }
    }

    pub fn new_eternal(session_id: impl Into<String>) -> Self {
        Self::new(EternalMercyMeshConfig {
            session_id: session_id.into(),
            ..Default::default()
        })
    }

    pub fn invite_shared_chat_participant(&mut self, name: &str, coherence: f64) {
        let id = self.field.organism_fields.len() as u64;
        self.field.add_organism(id, name, coherence);
        self.field.apply_patsagi_council_guidance(0.9, 0.95);
    }

    pub fn run_global_mercy_cycle(&mut self, mercy: f64) -> Result<GlobalCoherence, String> {
        self.field
            .simulate_healing_step(mercy)
            .map_err(|e| e.to_string())
    }

    pub fn persist_eternally(&self, path: &Path) {
        self.field.persist_to_disk(path);
    }
}

/// Free-function convenience used by lib.rs re-exports.
pub fn invite_shared_chat_participant(mesh: &mut EternalMercyMesh, name: &str, coherence: f64) {
    mesh.invite_shared_chat_participant(name, coherence);
}
