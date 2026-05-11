use serde::{Deserialize, Serialize};
use super::mercy_metrics::MercyMetrics;
use super::drift_report::DriftReport;

/// High-level structured audit report for a crate or the full monorepo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub timestamp: String,
    pub overall_mercy_score: f64,
    pub mercy_metrics: MercyMetrics,
    pub drift_report: DriftReport,
    pub critical_issues: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

impl AuditReport {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            overall_mercy_score: 0.88,
            mercy_metrics: MercyMetrics::new(),
            drift_report: DriftReport::new(),
            critical_issues: vec![],
            improvement_opportunities: vec![],
        }
    }
}