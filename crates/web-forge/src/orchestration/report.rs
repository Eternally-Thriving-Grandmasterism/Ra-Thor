/// Automated Reporting Module
///
/// Supports CI pipeline integration with JSON output and quality gates.

use crate::orchestration::advanced_orchestrator::AdvancedOrchestrationResult;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
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
            duration_seconds: None,
            wcag_aa_score: wcag.map(|s| s.score),
            wcag_aa_grade: wcag.map(|s| s.grade.clone()),
            validation_issues_count: result.validation_issues.len(),
            has_accessibility_issues: result.validation_issues.iter()
                .any(|i| i.to_lowercase().contains("accessibility") || i.contains("alt") || i.contains("label")),
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

    /// Quality gate for CI pipelines.
    /// Returns true if the result meets minimum acceptable standards.
    pub fn passes_ci_gate(&self, min_wcag_score: f32) -> bool {
        if !self.success {
            return false;
        }
        if let Some(score) = self.wcag_aa_score {
            if score < min_wcag_score {
                return false;
            }
        }
        true
    }

    /// Serialize report to pretty JSON (useful for CI artifacts).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}
