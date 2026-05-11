//! # Ra-Thor Plasticity Engine v2
//!
//! Safe, mercy-gated plasticity engine for self-evolution.

pub mod epigenetic_updater;
pub mod plasticity_rules;
pub mod safe_plasticity_applicator;

pub use epigenetic_updater::EpigeneticUpdater;
pub use plasticity_rules::{PlasticityRulesEngine, RuleResult};
pub use safe_plasticity_applicator::{SafePlasticityApplicator, RollbackPlan};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlasticityError {
    #[error("Invalid sensor data: {0}")]
    InvalidSensorData(String),
    #[error("Plasticity rule violation: {0}")]
    RuleViolation(String),
    #[error("Mercy Gate violation: {0}")]
    MercyGateViolation(String),
    #[error("Legal Lattice integration error: {0}")]
    LegalLatticeError(String),
}
