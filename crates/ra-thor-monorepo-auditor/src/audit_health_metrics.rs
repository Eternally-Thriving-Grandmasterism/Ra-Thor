//! Audit Health Metrics
//!
//! Provides observability into the quality and trends of audits performed by ra-thor-monorepo-auditor.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditHealthMetrics {
    pub total_audits: u64,
    pub avg_severity: f64,
    pub high_severity_count: u64,
    pub signals_triggering_proposals: u64,
    pub mercy_alignment_trend: String, // "improving", "stable", "degrading"
}

impl AuditHealthMetrics {
    pub fn new() -> Self {
        Self {
            mercy_alignment_trend: "stable".to_string(),
            ..Default::default()
        }
    }

    pub fn record_audit(&mut self, severity: f64, triggered_proposal: bool) {
        self.total_audits += 1;
        
        // Simple exponential moving average for severity
        let alpha = 0.1;
        self.avg_severity = alpha * severity + (1.0 - alpha) * self.avg_severity;

        if severity > 0.75 {
            self.high_severity_count += 1;
        }

        if triggered_proposal {
            self.signals_triggering_proposals += 1;
        }
    }

    pub fn get_health_score(&self) -> f64 {
        if self.total_audits == 0 {
            return 1.0;
        }
        let high_severity_ratio = self.high_severity_count as f64 / self.total_audits as f64;
        (1.0 - high_severity_ratio).max(0.0)
    }
}