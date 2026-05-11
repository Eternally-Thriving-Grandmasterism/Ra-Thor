//! # AuditSignal
//!
//! Rich, typed signals produced by ra-thor-monorepo-auditor for consumption
//! by ra-thor-meta-intelligence.

use serde::{Deserialize, Serialize};

/// A structured signal from the monorepo auditor.
/// Contains high-signal information about drift, mercy alignment, and improvement opportunities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSignal {
    pub signal_type: AuditSignalType,
    pub severity: f64,           // 0.0 - 1.0
    pub description: String,
    pub affected_crate: Option<String>,
    pub mercy_impact: f64,       // Estimated mercy alignment change if addressed
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditSignalType {
    OutdatedPattern,
    BrokenDependency,
    LowMercyCode,
    HallucinationRisk,
    DocumentationDrift,
    MissingTestCoverage,
    PlasticityOpportunity,
    SelfImprovementReady,
}
