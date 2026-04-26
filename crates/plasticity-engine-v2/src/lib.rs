//! Ra-Thor Plasticity Engine v2
//! Real-time epigenetic plasticity updates for the 5-Gene Joy Tetrad
//! Integrates with Legal Lattice, Sensor Fusion Bridge, and 200-year mercy legacy

pub mod epigenetic_updater;
pub mod plasticity_rules;

pub use epigenetic_updater::{EpigeneticUpdater, PlasticityUpdate};
pub use plasticity_rules::{PlasticityRule, PlasticityRulesEngine};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlasticityError {
    #[error("Invalid sensor data: {0}")]
    InvalidSensorData(String),
    
    #[error("Plasticity rule violation: {0}")]
    RuleViolation(String),
}

pub struct PlasticityEngineV2 {
    updater: epigenetic_updater::EpigeneticUpdater,
    rules: plasticity_rules::PlasticityRulesEngine,
}

impl PlasticityEngineV2 {
    pub fn new() -> Self {
        Self {
            updater: epigenetic_updater::EpigeneticUpdater::new(),
            rules: plasticity_rules::PlasticityRulesEngine::new(),
        }
    }

    pub async fn process_daily_update(
        &self,
        sensor_data: &crate::legal_lattice::sensor_fusion_bridge::MercyGelReading,
    ) -> Result<PlasticityUpdate, PlasticityError> {
        let cehi_impact = self.updater.calculate_from_sensors(sensor_data).await?;
        
        let rule_result = self.rules.evaluate(&cehi_impact).await?;
        
        if rule_result.should_apply() {
            self.updater.apply_plasticity_update(&cehi_impact, &rule_result).await?;
        }
        
        Ok(PlasticityUpdate {
            cehi_delta: cehi_impact.improvement,
            rule_applied: rule_result.rule_name,
            timestamp: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PlasticityUpdate {
    pub cehi_delta: f64,
    pub rule_applied: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
