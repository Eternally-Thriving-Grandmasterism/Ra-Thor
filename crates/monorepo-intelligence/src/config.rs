//! # Configuration System (v0.3.0)
//!
//! TOML-based, hot-reloadable configuration for the Monorepo Intelligence system.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceConfig {
    pub scanner: ScannerConfig,
    pub search: SearchConfig,
    pub reporting: ReportingConfig,
    pub health: HealthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerConfig {
    pub include_hidden: bool,
    pub max_depth: Option<usize>,
    pub enable_parallel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub enable_semantic_scoring: bool,
    pub max_results: usize,
    pub relevance_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    pub default_format: String, // "markdown", "html", "json"
    pub include_health_score: bool,
    pub generate_recommendations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub powrush_weight: f32,
    pub documentation_weight: f32,
    pub crate_structure_weight: f32,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            scanner: ScannerConfig {
                include_hidden: false,
                max_depth: None,
                enable_parallel: true,
            },
            search: SearchConfig {
                enable_semantic_scoring: true,
                max_results: 100,
                relevance_threshold: 0.3,
            },
            reporting: ReportingConfig {
                default_format: "markdown".to_string(),
                include_health_score: true,
                generate_recommendations: true,
            },
            health: HealthConfig {
                powrush_weight: 0.35,
                documentation_weight: 0.30,
                crate_structure_weight: 0.35,
            },
        }
    }
}

impl IntelligenceConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        toml::from_str(&content).map_err(|e| e.to_string())
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let content = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, content).map_err(|e| e.to_string())
    }
}
