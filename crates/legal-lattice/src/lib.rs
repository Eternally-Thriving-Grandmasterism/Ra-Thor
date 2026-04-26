//! Ra-Thor Legal Lattice Governance Engine
//! 
//! Native integration of the full legal stack (28th Amendment + Mercy-Gated Acts + International Treaty)
//! into the Ra-Thor living lattice as a first-class, enforcement-ready layer.

pub mod amendment28;
pub mod gates;
pub mod cehi;
pub mod fund;
pub mod sensor_fusion_bridge;

pub use amendment28::{Amendment28Validator, Amendment28Violation};
pub use gates::{GatesEngine, GatesViolation};
pub use cehi::{FiveGeneCEHI, CEHICalculator, GeneMethylation, CEHIImpact, DisbursementTier};
pub use fund::{MercyLegacyFundEngine, TrancheRelease};
pub use sensor_fusion_bridge::{SensorFusionBridge, MercyGelReading};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LegalLatticeError {
    #[error("28th Amendment violation: {0}")]
    Amendment28Violation(#[from] Amendment28Violation),
    
    #[error("7 Gates validation failed: {0}")]
    GatesViolation(#[from] GatesViolation),
    
    #[error("CEHI calculation error: {0}")]
    CEHIError(String),
    
    #[error("Mercy Legacy Fund disbursement error: {0}")]
    FundError(String),
}

/// The main Legal Lattice Governance Engine
pub struct LegalLatticeEngine {
    amendment28: amendment28::Amendment28Validator,
    gates: gates::GatesEngine,
    cehi: cehi::CEHICalculator,
    fund: fund::MercyLegacyFundEngine,
}

impl LegalLatticeEngine {
    pub fn new() -> Self {
        Self {
            amendment28: amendment28::Amendment28Validator::new(),
            gates: gates::GatesEngine::new(),
            cehi: cehi::CEHICalculator::new(),
            fund: fund::MercyLegacyFundEngine::new(),
        }
    }

    /// Validates any proposed action against the full legal stack
    pub async fn validate_action(&self, action: &Action) -> Result<ValidatedAction, LegalLatticeError> {
        // 1. Hard 28th Amendment check (non-bypassable)
        self.amendment28.validate(action)?;
        
        // 2. Real-time 7 Gates validation
        let gates_result = self.gates.validate(action).await?;
        
        // 3. Calculate 5-gene CEHI impact
        let cehi_impact = self.cehi.calculate_impact(action).await?;
        
        // 4. Route to Mercy Legacy Fund if eligible
        if cehi_impact.meets_disbursement_threshold() {
            self.fund.queue_disbursement(cehi_impact.clone()).await?;
        }
        
        Ok(ValidatedAction {
            gates_result,
            cehi_impact,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Validates action using real-time sensor data via the Sensor Fusion Bridge
    pub async fn validate_with_sensors(
        &self,
        action: &Action,
        sensor_reading: &sensor_fusion_bridge::MercyGelReading,
    ) -> Result<ValidatedAction, LegalLatticeError> {
        let bridge = sensor_fusion_bridge::SensorFusionBridge::new();
        let cehi_impact = bridge.calculate_impact_from_sensors(sensor_reading, 3.85);

        // Still run 28th Amendment and 7 Gates checks
        self.amendment28.validate(action)?;
        let gates_result = self.gates.validate(action).await?;

        if cehi_impact.meets_disbursement_threshold() {
            self.fund.queue_disbursement(cehi_impact.clone()).await?;
        }

        Ok(ValidatedAction {
            gates_result,
            cehi_impact,
            timestamp: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: String,
    pub biological_impact: bool,
    pub cehi_contribution: Option<f64>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ValidatedAction {
    pub gates_result: gates::GatesResult,
    pub cehi_impact: cehi::CEHIImpact,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
