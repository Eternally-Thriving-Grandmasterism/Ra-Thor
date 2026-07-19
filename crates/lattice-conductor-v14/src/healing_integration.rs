//! HealingFieldRegistry integration for LatticeConductorV14 (v14.8.2)
//! Complete minimal production surface — no incomplete stubs.

use crate::clifford_healing_fields::{
    CliffordHealingField, GlobalCoherence, HealingConfig, HealingFieldError,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HealingTelemetry {
    pub last_coherence: f64,
    pub cycles_run: u64,
    pub last_organism_count: usize,
}

impl Default for HealingTelemetry {
    fn default() -> Self {
        Self {
            last_coherence: 1.0,
            cycles_run: 0,
            last_organism_count: 0,
        }
    }
}

pub struct HealingFieldRegistry {
    pub fields: HashMap<String, CliffordHealingField>,
    pub global_config: HealingConfig,
    pub telemetry: HealingTelemetry,
}

impl HealingFieldRegistry {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            global_config: HealingConfig::default(),
            telemetry: HealingTelemetry::default(),
        }
    }

    pub fn register_field(&mut self, name: &str) -> &mut CliffordHealingField {
        self.fields
            .entry(name.to_string())
            .or_insert_with(|| CliffordHealingField::new(name))
    }

    /// Run a global healing cycle across all registered fields.
    pub fn run_global_healing_cycle(
        &mut self,
        mercy: f64,
    ) -> Result<GlobalCoherence, HealingFieldError> {
        let mut total_mercy = 0.0;
        let mut total_organisms = 0usize;
        let mut max_step = 0u64;

        for field in self.fields.values_mut() {
            let coherence = field.simulate_healing_step(mercy)?;
            total_mercy += coherence.average_mercy * coherence.organism_count as f64;
            total_organisms += coherence.organism_count;
            max_step = max_step.max(coherence.evolution_step);
        }

        let average_mercy = if total_organisms == 0 {
            1.0
        } else {
            total_mercy / total_organisms as f64
        };

        self.telemetry.last_coherence = average_mercy;
        self.telemetry.cycles_run += 1;
        self.telemetry.last_organism_count = total_organisms;

        Ok(GlobalCoherence {
            average_mercy,
            organism_count: total_organisms,
            evolution_step: max_step,
        })
    }
}

impl Default for HealingFieldRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Free-function convenience matching lib.rs re-export.
pub fn run_global_healing_cycle(
    registry: &mut HealingFieldRegistry,
    mercy: f64,
) -> Result<GlobalCoherence, HealingFieldError> {
    registry.run_global_healing_cycle(mercy)
}
