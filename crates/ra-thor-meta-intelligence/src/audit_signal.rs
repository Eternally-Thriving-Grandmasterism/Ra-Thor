//! AuditSignal
//!
//! Structured signals from ra-thor-monorepo-auditor that feed into
//! ra-thor-meta-intelligence for mercy-gated self-improvement decisions.

use serde::{Serialize, Deserialize};

/// Structured signal produced by the monorepo auditor.
/// This replaces raw string signals with typed, rich context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditSignal {
    /// Significant code or documentation drift detected
    DriftDetected {
        crate_name: String,
        severity: f64,
        description: String,
    },

    /// Mercy alignment issue or low-valence pattern found
    MercyAlignmentIssue {
        location: String,
        current_valence: f64,
        description: String,
    },

    /// TOLC inconsistency or ethical constraint violation detected
    TolcInconsistency {
        area: String,
        severity: f64,
        description: String,
    },

    /// General outdated pattern or technical debt
    OutdatedPattern {
        crate_name: String,
        pattern_type: String,
        description: String,
    },

    /// Positive signal (high mercy, strong TOLC, good health)
    PositiveHealthSignal {
        area: String,
        score: f64,
        description: String,
    },
}

impl AuditSignal {
    pub fn is_critical(&self) -> bool {
        match self {
            AuditSignal::DriftDetected { severity, .. } => *severity > 0.7,
            AuditSignal::MercyAlignmentIssue { current_valence, .. } => *current_valence < 0.85,
            AuditSignal::TolcInconsistency { severity, .. } => *severity > 0.65,
            _ => false,
        }
    }

    pub fn mercy_impact_score(&self) -> f64 {
        match self {
            AuditSignal::MercyAlignmentIssue { current_valence, .. } => 1.0 - current_valence,
            AuditSignal::DriftDetected { severity, .. } => *severity * 0.8,
            AuditSignal::TolcInconsistency { severity, .. } => *severity * 0.9,
            AuditSignal::OutdatedPattern { .. } => 0.6,
            AuditSignal::PositiveHealthSignal { score, .. } => *score * 0.3,
        }
    }
}
