//! # Monorepo Health Scoring System (v0.2.0)
//!
//! Advanced, multi-dimensional health scoring for any module (especially Powrush).

use crate::scanner::ScanResult;

#[derive(Debug, Clone)]
pub struct MonorepoHealthScore {
    pub overall_score: f32,           // 0.0 – 100.0
    pub powrush_score: f32,
    pub crate_structure_score: f32,
    pub documentation_score: f32,
    pub integration_score: f32,
    pub recommendations: Vec<String>,
}

impl MonorepoHealthScore {
    pub fn calculate(scan: &ScanResult, focus_module: &str) -> Self {
        let powrush_files = scan.files.iter()
            .filter(|f| f.relative_path.to_lowercase().contains("powrush"))
            .count();

        let powrush_score = ((powrush_files as f32 / 60.0) * 100.0).min(100.0);

        let crate_dirs = scan.files.iter()
            .filter(|f| f.relative_path.contains("crates/") && f.is_dir)
            .count();

        let crate_structure_score = ((crate_dirs as f32 / 18.0) * 100.0).min(100.0);

        let doc_files = scan.files.iter()
            .filter(|f| f.relative_path.contains("docs/"))
            .count();

        let documentation_score = ((doc_files as f32 / 90.0) * 100.0).min(100.0);

        let integration_score = if powrush_score > 70.0 && crate_structure_score > 60.0 {
            85.0
        } else {
            55.0
        };

        let overall = (powrush_score + crate_structure_score + documentation_score + integration_score) / 4.0;

        let mut recommendations = vec![];

        if powrush_score < 65.0 {
            recommendations.push("Consolidate all Powrush files into a single unified `crates/powrush` crate.".to_string());
        }
        if documentation_score < 70.0 {
            recommendations.push("Expand documentation coverage for better AI understanding and onboarding.".to_string());
        }
        if crate_structure_score < 60.0 {
            recommendations.push("Improve crate organization and modularity.".to_string());
        }

        Self {
            overall_score: overall,
            powrush_score,
            crate_structure_score,
            documentation_score,
            integration_score,
            recommendations,
        }
    }
}
