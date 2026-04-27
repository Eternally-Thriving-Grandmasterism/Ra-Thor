//! # Monorepo Health Scoring System (v0.3.0)
//!
//! Advanced multi-dimensional health scoring for the entire monorepo.
//! Especially powerful for Powrush and strategic modules.

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
        // Powrush-specific scoring
        let powrush_files = scan.files.iter()
            .filter(|f| f.relative_path.to_lowercase().contains("powrush"))
            .count();

        let powrush_score = ((powrush_files as f32 / 65.0) * 100.0).min(100.0);

        // Crate structure scoring
        let crate_dirs = scan.files.iter()
            .filter(|f| f.relative_path.contains("crates/") && f.is_dir)
            .count();

        let crate_structure_score = ((crate_dirs as f32 / 20.0) * 100.0).min(100.0);

        // Documentation scoring
        let doc_files = scan.files.iter()
            .filter(|f| f.relative_path.contains("docs/"))
            .count();

        let documentation_score = ((doc_files as f32 / 95.0) * 100.0).min(100.0);

        // Integration scoring (how well things connect)
        let integration_score = if powrush_score > 65.0 && crate_structure_score > 60.0 {
            88.0
        } else if powrush_score > 50.0 {
            72.0
        } else {
            55.0
        };

        // Overall weighted score
        let overall = (powrush_score * 0.35 +
                       crate_structure_score * 0.30 +
                       documentation_score * 0.20 +
                       integration_score * 0.15).min(100.0);

        // Generate smart recommendations
        let mut recommendations = vec![];

        if powrush_score < 65.0 {
            recommendations.push("Consolidate all Powrush files into a single unified `crates/powrush` structure.".to_string());
        }
        if documentation_score < 70.0 {
            recommendations.push("Expand documentation coverage for better AI understanding and onboarding.".to_string());
        }
        if crate_structure_score < 60.0 {
            recommendations.push("Improve crate organization and modularity across the monorepo.".to_string());
        }
        if overall < 75.0 {
            recommendations.push("Run the full Monorepo Intelligence health check and address top recommendations.".to_string());
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
