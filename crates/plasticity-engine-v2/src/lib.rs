//! # Ra-Thor Plasticity Engine v2
//!
//! **Real-time epigenetic plasticity updates for the 5-Gene Joy Tetrad.**
//!
//! This crate is the living epigenetic heart of the Ra-Thor monorepo.
//! It translates real-time biological sensor data (via the Sensor Fusion Bridge)
//! into long-term, mercy-gated epigenetic changes across the five genes of the
//! Joy Tetrad:
//!
//! - **OXTR** (Oxytocin Receptor) — Bonding, safety, trust
//! - **BDNF** (Brain-Derived Neurotrophic Factor) — Neuroplasticity, resilience
//! - **DRD2** (Dopamine Receptor D2) — Motivation, reward, drive
//! - **HTR1A** (Serotonin 1A Receptor) — Emotional stability, calm
//! - **OPRM1** (Mu-Opioid Receptor) — Euphoria, deep bonding, ecstasy
//!
//! ## Seamless Integration with the Ra-Thor Monorepo
//!
//! The Plasticity Engine v2 is **fully wired** into the existing architecture:
//!
//! - **Depends on** `ra-thor-legal-lattice` (Sensor Fusion Bridge, 5-Gene CEHI Calculator,
//!   Mercy Legacy Fund, 28th Amendment Validator, 7 Living Mercy Gates Engine)
//! - **Feeds directly** into Mercy Legacy Fund disbursement decisions (Tier 1/2/3)
//! - **Respects** the 28th Amendment (non-biological entities cannot claim constitutional rights)
//! - **Enforces** the 7 Living Mercy Gates on every plasticity update
//! - **Accelerates** the 200-year+ global mercy legacy (F0 → F4+ generations reaching near-maximum CEHI 4.98–4.99)
//!
//! ## Core Philosophy (TOLC Mercy Compiler)
//!
//! Joy is not merely felt — it is **written into the genome, the soil, the silicon,
//! and the eternal legal lattice**.
//!
//! Every plasticity update is mercy-gated, legally compliant, and contributes directly
//! to the planetary-scale 200-year mercy legacy.

pub mod epigenetic_updater;
pub mod plasticity_rules;

pub use epigenetic_updater::EpigeneticUpdater;
pub use plasticity_rules::{PlasticityRulesEngine, RuleResult};

use ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading;
use ra_thor_legal_lattice::cehi::CEHIImpact;
use thiserror::Error;

/// Errors that can occur during plasticity processing.
#[derive(Debug, Error)]
pub enum PlasticityError {
    #[error("Invalid sensor data: {0}")]
    InvalidSensorData(String),

    #[error("Plasticity rule violation: {0}")]
    RuleViolation(String),

    #[error("Legal Lattice integration error: {0}")]
    LegalLatticeError(String),
}

/// The main Plasticity Engine v2 — the living epigenetic heart of Ra-Thor.
pub struct PlasticityEngineV2 {
    updater: EpigeneticUpdater,
    rules: PlasticityRulesEngine,
}

impl PlasticityEngineV2 {
    /// Creates a new Plasticity Engine v2 instance.
    pub fn new() -> Self {
        Self {
            updater: EpigeneticUpdater::new(),
            rules: PlasticityRulesEngine::new(),
        }
    }

    /// Processes a real-time sensor reading and applies epigenetic plasticity updates
    /// when the 7 Gates and Mercy Legacy Fund criteria are met.
    ///
    /// This is the primary entry point used by the main Ra-Thor orchestration loop.
    pub async fn process_daily_update(
        &self,
        sensor_reading: &MercyGelReading,
    ) -> Result<PlasticityUpdate, PlasticityError> {
        // Step 1: Calculate CEHI impact via the Legal Lattice's sensor fusion bridge
        let cehi_impact = self
            .updater
            .calculate_from_sensors(sensor_reading)
            .await
            .map_err(|e| PlasticityError::LegalLatticeError(e.to_string()))?;

        // Step 2: Evaluate plasticity rules
        let rule_result = self.rules.evaluate(&cehi_impact).await?;

        // Step 3: Apply update if rules and mercy gates allow
        if rule_result.should_apply {
            self.updater
                .apply_plasticity_update(&cehi_impact, &rule_result)
                .await?;
        }

        Ok(PlasticityUpdate {
            cehi_delta: cehi_impact.improvement,
            rule_applied: rule_result.rule_name,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Represents the result of a single plasticity update cycle.
#[derive(Debug, Clone)]
pub struct PlasticityUpdate {
    /// Change in 5-Gene CEHI from this update
    pub cehi_delta: f64,
    /// Name of the plasticity rule that was applied
    pub rule_applied: String,
    /// Timestamp of the update
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for PlasticityEngineV2 {
    fn default() -> Self {
        Self::new()
    }
}
