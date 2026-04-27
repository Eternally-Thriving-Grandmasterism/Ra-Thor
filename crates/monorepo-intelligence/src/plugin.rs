//! # Plugin System (v0.3.0)
//!
//! Trait-based extensibility. Any AI or developer can create custom analyzers.

use crate::scanner::ScanResult;
use async_trait::async_trait;

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

/// Example built-in plugin: Powrush Focus Analyzer
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

        let score = (powrush_count as f32 / 50.0 * 100.0).min(100.0);

        Ok(PluginResult {
            name: self.name().to_string(),
            score,
            message: format!("Powrush coverage: {:.1}%", score),
            recommendations: if score < 70.0 {
                vec!["Consolidate Powrush into a unified crate structure.".to_string()]
            } else {
                vec![]
            },
        })
    }
}
