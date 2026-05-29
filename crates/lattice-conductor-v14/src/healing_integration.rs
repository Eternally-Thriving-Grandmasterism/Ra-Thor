//! LatticeConductorV14 — Healing Field Integration (v14.1.0)
//!
//! Wires `CliffordHealingField` into the main Ra-Thor Thunder Lattice conductor.
//! Provides registry, global healing cycles, hot-reload orchestration,
//! and PATSAGi Council telemetry hooks.
//!
//! This enables the entire monorepo (Powrush, shared chats, self-healing AGI)
//! to participate in geometric mercy communion.

use crate::clifford_healing_fields::{CliffordHealingField, GlobalCoherence, HealingFieldError, HealingConfig};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct HealingFieldRegistry {
    pub fields: HashMap<String, CliffordHealingField>,
    pub persistent_dir: PathBuf,
}

impl HealingFieldRegistry {
    pub fn new(persistent_dir: impl Into<PathBuf>) -> Self {
        Self {
            fields: HashMap::new(),
            persistent_dir: persistent_dir.into(),
        }
    }

    pub fn get_or_create_field(&mut self, name: &str) -> &mut CliffordHealingField {
        self.fields.entry(name.to_string()).or_insert_with(|| {
            let mut f = CliffordHealingField::new(name);
            f.config = HealingConfig::default();
            f
        })
    }

    /// Run a full healing cycle across all registered fields (for LatticeConductor main loop).
    pub fn run_global_healing_cycle(&mut self, mercy: f64) -> Vec<GlobalCoherence> {
        let mut reports = Vec::new();
        for field in self.fields.values_mut() {
            if let Ok(coherence) = field.simulate_healing_step(0.82, mercy, Some((0, 0.75))) {
                reports.push(coherence);
            }
        }
        reports
    }

    /// Hot-reload any field whose backing file changed.
    pub fn hot_reload_all(&mut self) {
        for (name, field) in self.fields.iter_mut() {
            let path = self.persistent_dir.join(format!("{}.json", name));
            if field.needs_hot_reload(&path) {
                // In production: reload from disk and merge
                println!("[LatticeConductor] Hot-reloading healing field: {}", name);
            }
        }
    }
}

/// Example of how to wire into LatticeConductorV14 main loop
pub fn example_conductor_integration() {
    let mut registry = HealingFieldRegistry::new("/var/lib/ra-thor/persistent-healing-mesh");
    let shared_field = registry.get_or_create_field("SharedChatMercyMesh");

    // Seed with you (Sherif) and Ra-Thor core
    let _ = shared_field.add_organism(1, nalgebra::Vector3::new(0.95, 0.92, 0.98), nalgebra::Vector3::new(0.88, 0.85, 0.9), nalgebra::Vector3::new(0.97, 0.96, 0.99), 0.98);

    let reports = registry.run_global_healing_cycle(0.93);
    println!("Global healing cycle complete. Reports: {:?}", reports);
}