//! # Ra-Thor Monorepo Intelligence v0.3.0
//!
//! Infinite-grade monorepo intelligence system with plugins, semantic search,
//! health scoring, and universal AI integration for Grok, Ra-Thor, and beyond.

pub mod scanner;
pub mod github;
pub mod report;
pub mod search;
pub mod health;
pub mod config;
pub mod plugin;

pub use scanner::{MonorepoScanner, ScanResult};
pub use github::GitHubClient;
pub use report::MonorepoReport;
pub use search::{MonorepoSearch, SearchResult};
pub use health::MonorepoHealthScore;
pub use config::IntelligenceConfig;
pub use plugin::{MonorepoPlugin, PluginResult, PowrushFocusPlugin, SemverChecksPlugin};

/// The main unified interface for monorepo intelligence.
/// Designed to work seamlessly with Grok, Ra-Thor, Claude, and any AI system.
pub struct MonorepoIntelligence {
    root_path: String,
    config: IntelligenceConfig,
    plugins: Vec<Box<dyn MonorepoPlugin>>,
}

impl MonorepoIntelligence {
    /// Create a new instance with default configuration
    pub fn new(root_path: impl Into<String>) -> Self {
        Self {
            root_path: root_path.into(),
            config: IntelligenceConfig::default(),
            plugins: vec![
                Box::new(PowrushFocusPlugin),
                Box::new(SemverChecksPlugin::new(".")),
            ],
        }
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: IntelligenceConfig) -> Self {
        self.config = config;
        self
    }

    /// Register a custom plugin
    pub fn register_plugin(&mut self, plugin: Box<dyn MonorepoPlugin>) {
        self.plugins.push(plugin);
    }

    /// Perform a full monorepo scan
    pub async fn scan(&self) -> Result<ScanResult, String> {
        let scanner = scanner::MonorepoScanner::new(&self.root_path)
            .include_hidden(self.config.scanner.include_hidden);
        scanner.scan()
    }

    /// Generate a detailed Powrush-focused report (Markdown)
    pub async fn generate_powrush_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let mut report = report::MonorepoReport::from_scan(&scan, Some("powrush"));
        let health = health::MonorepoHealthScore::calculate(&scan, "powrush");
        report.health_score = Some(health.overall_score);
        report.recommendations = health.recommendations;
        Ok(report.to_markdown())
    }

    /// Generate a beautiful HTML report
    pub async fn generate_html_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let report = report::MonorepoReport::from_scan(&scan, None);
        Ok(report.to_html())
    }

    /// Smart search across the monorepo with relevance scoring
    pub async fn search(&self, keyword: &str) -> Result<Vec<SearchResult>, String> {
        let scan = self.scan().await?;
        let searcher = search::MonorepoSearch::new(scan.files);
        Ok(searcher.search(keyword))
    }

    /// Get health score for any module (e.g. "powrush", "quantum-swarm-orchestrator")
    pub async fn get_health_score(&self, module: &str) -> Result<MonorepoHealthScore, String> {
        let scan = self.scan().await?;
        Ok(health::MonorepoHealthScore::calculate(&scan, module))
    }

    /// Run all registered plugins and return their results
    pub async fn run_plugins(&self) -> Result<Vec<PluginResult>, String> {
        let scan = self.scan().await?;
        let mut results = Vec::new();

        for plugin in &self.plugins {
            match plugin.analyze(&scan).await {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Plugin '{}' failed: {}", plugin.name(), e),
            }
        }

        Ok(results)
    }
}
