use serde::{Deserialize, Serialize};

/// Structured mercy-aligned metrics for a crate or the full monorepo.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MercyMetrics {
    /// Current valence / positive emotion score (0.0 – 1.0)
    pub valence: f64,
    /// How well the code aligns with the 7 Living Mercy Gates
    pub mercy_alignment: f64,
    /// How ready the system is for safe self-improvement
    pub self_improvement_readiness: f64,
    /// Number of detected low-mercy or high-risk patterns
    pub low_mercy_violations: u32,
}

impl MercyMetrics {
    pub fn new() -> Self {
        Self {
            valence: 0.85,
            mercy_alignment: 0.90,
            self_improvement_readiness: 0.75,
            low_mercy_violations: 0,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.valence >= 0.8 && self.mercy_alignment >= 0.85
    }
}