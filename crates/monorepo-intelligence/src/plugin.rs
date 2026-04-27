//! # Plugin System (v0.3.0)
//!
//! Trait-based extensibility for Ra-Thor Monorepo Intelligence.
//! Includes built-in plugins and support for cargo-semver-checks integration.

use crate::scanner::ScanResult;
use async_trait::async_trait;
use std::process::Command;

#[async_trait]
pub trait MonorepoPlugin: Send + Sync {
    fn name(&self) -> &'static str;
    async fn analyze(&self, scan: &ScanResult) -> Result<PluginResult, String>;
}

#[derive(Debug, Clone)]
pub struct PluginResult {
    pub name: String,
    pub score: f32,
    pub message: String,
    pub recommendations: Vec<String>,
}

/// Built-in plugin: Powrush Focus Analyzer
pub struct PowrushFocusPlugin;

#[async_trait]
impl MonorepoPlugin for PowrushFocusPlugin {
    fn name(&self) -> &'static str {
        "Powrush Focus Analyzer"
    }

    async fn analyze(&self, scan: &ScanResult) -> Result<PluginResult, String> {
        let powrush_count = scan.files.iter()
            .filter(|f| f.relative_path.to_lowercase().contains("powrush"))
            .count();

        let score = (powrush_count as f32 / 60.0 * 100.0).min(100.0);

        let mut recommendations = vec![];
        if score < 70.0 {
            recommendations.push("Consolidate all Powrush files into a single `crates/powrush` structure.".to_string());
        }

        Ok(PluginResult {
            name: self.name().to_string(),
            score,
            message: format!("Powrush coverage: {:.1}%", score),
            recommendations,
        })
    }
}

/// Plugin: cargo-semver-checks Integration
pub struct SemverChecksPlugin {
    pub crate_path: String,
}

impl SemverChecksPlugin {
    pub fn new(crate_path: impl Into<String>) -> Self {
        Self {
            crate_path: crate_path.into(),
        }
    }
}

#[async_trait]
impl MonorepoPlugin for SemverChecksPlugin {
    fn name(&self) -> &'static str {
        "cargo-semver-checks Integration"
    }

    async fn analyze(&self, _scan: &ScanResult) -> Result<PluginResult, String> {
        let output = Command::new("cargo")
            .args(["semver-checks", "--package", &self.crate_path])
            .output()
            .map_err(|e| format!("Failed to run cargo-semver-checks: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (score, message, recommendations) = if output.status.success() {
            (100.0, "No semver violations detected".to_string(), vec![])
        } else {
            let violations = stderr.lines()
                .filter(|l| l.contains("breaking") || l.contains("error"))
                .count();

            let score = (100.0 - (violations as f32 * 8.0)).max(0.0);
            let mut recs = vec!["Review and fix semver breaking changes.".to_string()];

            if violations > 5 {
                recs.push("Consider a major version bump.".to_string());
            }

            (score, format!("Detected {} semver issues", violations), recs)
        };

        Ok(PluginResult {
            name: self.name().to_string(),
            score,
            message,
            recommendations,
        })
    }
}
