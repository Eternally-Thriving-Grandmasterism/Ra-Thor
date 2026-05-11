// crates/ra-thor-meta-intelligence/src/crate_analyzer.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateHealthReport {
    pub crate_name: String,
    pub importance: f64,
    pub mercy_integration_score: f64,
    pub technical_debt_score: f64,
    pub test_coverage_score: f64,
    pub documentation_score: f64,
    pub overall_health: f64,

    // Phase 4.3 enhancements
    pub self_improvement_potential: f64,
    pub risk_level: RiskLevel,
    pub last_analyzed: Option<DateTime<Utc>>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

pub struct CrateAnalyzer {
}

impl CrateAnalyzer {
    pub fn new() -> Self {
        Self {}
    }

    /// Analyzes a single crate using lightweight but real file-based heuristics.
    /// This is the foundation for more advanced analysis in later phases.
    pub fn analyze_crate_basic_health(&self, crate_path: &Path) -> CrateHealthReport {
        let crate_name = crate_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let has_src = crate_path.join("src").exists();
        let has_tests = crate_path.join("tests").exists() || crate_path.join("src").join("tests").exists();
        let has_readme = crate_path.join("README.md").exists();
        let has_cargo_toml = crate_path.join("Cargo.toml").exists();

        let documentation_score = if has_readme { 0.85 } else { 0.35 };
        let test_coverage_score = if has_tests { 0.75 } else { 0.25 };
        let structural_score = if has_src && has_cargo_toml { 0.90 } else { 0.50 };

        let overall_health = (documentation_score + test_coverage_score + structural_score) / 3.0;

        // Higher potential when health is lower (more room to improve)
        let self_improvement_potential = (1.0 - overall_health).clamp(0.1, 0.95);

        let risk_level = if overall_health < 0.5 {
            RiskLevel::High
        } else if overall_health < 0.75 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        let mut notes = Vec::new();
        if !has_readme {
            notes.push("Missing README.md — documentation is weak".to_string());
        }
        if !has_tests {
            notes.push("No dedicated tests directory found".to_string());
        }
        if !has_src {
            notes.push("Missing src/ directory — unusual structure".to_string());
        }

        CrateHealthReport {
            crate_name,
            importance: 0.70,
            mercy_integration_score: 0.80,
            technical_debt_score: 1.0 - overall_health,
            test_coverage_score,
            documentation_score,
            overall_health,
            self_improvement_potential,
            risk_level,
            last_analyzed: Some(Utc::now()),
            notes,
        }
    }

    // Existing methods kept for compatibility (can be expanded later)
    pub async fn analyze_critical_crates(&self) -> Result<Vec<CrateHealthReport>, Box<dyn std::error::Error + Send + Sync>> {
        // Placeholder - can be updated to use analyze_crate_basic_health in future
        Ok(vec![])
    }
}