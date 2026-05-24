/// Automated Reporting Module
///
/// Generates structured reports from orchestration results.
/// Useful for CI/CD, dashboards, logging, and quality tracking.

use crate::orchestration::advanced_orchestrator::AdvancedOrchestrationResult;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct OrchestrationReport {
    pub success: bool,
    pub attempts: usize,
    pub duration_seconds: Option<f64>,
    pub wcag_aa_score: Option<f32>,
    pub wcag_aa_grade: Option<String>,
    pub validation_issues_count: usize,
    pub has_accessibility_issues: bool,
}

impl From<&AdvancedOrchestrationResult> for OrchestrationReport {
    fn from(result: &AdvancedOrchestrationResult) -> Self {
        let wcag = result.wcag_aa_score.as_ref();

        Self {
            success: result.success,
            attempts: result.attempts_used,
            duration_seconds: None, // Can be enriched later
            wcag_aa_score: wcag.map(|s| s.score),
            wcag_aa_grade: wcag.map(|s| s.grade.clone()),
            validation_issues_count: result.validation_issues.len(),
            has_accessibility_issues: result.validation_issues.iter().any(|i| i.to_lowercase().contains("accessibility") || i.contains("alt") || i.contains("label")),
        }
    }
}

impl OrchestrationReport {
    pub fn summary(&self) -> String {
        format!(
            "Success: {} | Attempts: {} | WCAG AA: {} ({}) | Issues: {}",
            self.success,
            self.attempts,
            self.wcag_aa_score.map_or("N/A".to_string(), |s| format!("{:.1}", s)),
            self.wcag_aa_grade.as_deref().unwrap_or("N/A"),
            self.validation_issues_count
        )
    }
}
