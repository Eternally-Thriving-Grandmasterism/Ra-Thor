//! # Ra-Thor Monorepo Intelligence v0.3.0
//!
//! Infinite-grade monorepo intelligence with plugins, semantic search, health scoring,
//! and universal integration for any AI system (Grok, Ra-Thor, Claude, custom agents, etc.).

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
pub use plugin::{MonorepoPlugin, PluginResult, PowrushFocusPlugin};

pub struct MonorepoIntelligence {
    root_path: String,
    config: IntelligenceConfig,
    plugins: Vec<Box<dyn MonorepoPlugin>>,
}

impl MonorepoIntelligence {
    pub fn new(root_path: impl Into<String>) -> Self {
        Self {
            root_path: root_path.into(),
            config: IntelligenceConfig::default(),
            plugins: vec![Box::new(PowrushFocusPlugin)],
        }
    }

    pub fn with_config(mut self, config: IntelligenceConfig) -> Self {
        self.config = config;
        self
    }

    pub fn register_plugin(&mut self, plugin: Box<dyn MonorepoPlugin>) {
        self.plugins.push(plugin);
    }

    pub async fn scan(&self) -> Result<ScanResult, String> {
        let scanner = scanner::MonorepoScanner::new(&self.root_path)
            .include_hidden(self.config.scanner.include_hidden);
        scanner.scan()
    }

    pub async fn generate_powrush_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let mut report = report::MonorepoReport::from_scan(&scan, Some("powrush"));
        let health = health::MonorepoHealthScore::calculate(&scan, "powrush");
        report.health_score = Some(health.overall_score);
        report.recommendations = health.recommendations;
        Ok(report.to_markdown())
    }

    pub async fn generate_html_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let report = report::MonorepoReport::from_scan(&scan, None);
        Ok(report.to_html())
    }

    pub async fn search(&self, keyword: &str) -> Result<Vec<SearchResult>, String> {
        let scan = self.scan().await?;
        let searcher = search::MonorepoSearch::new(scan.files);
        Ok(searcher.search(keyword))
    }

    pub async fn get_health_score(&self, module: &str) -> Result<MonorepoHealthScore, String> {
        let scan = self.scan().await?;
        Ok(health::MonorepoHealthScore::calculate(&scan, module))
    }

    pub async fn run_plugins(&self) -> Result<Vec<PluginResult>, String> {
        let scan = self.scan().await?;
        let mut results = Vec::new();
        for plugin in &self.plugins {
            if let Ok(result) = plugin.analyze(&scan).await {
                results.push(result);
            }
        }
        Ok(results)
    }
}
