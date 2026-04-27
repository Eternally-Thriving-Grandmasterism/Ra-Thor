//! # Ra-Thor Monorepo Intelligence v0.3.0
//!
//! Infinite-grade monorepo intelligence with plugins, semantic search, health scoring,
//! and universal integration for any AI system.

pub mod scanner;
pub mod github;
pub mod report;
pub mod search;
pub mod health;
pub mod config;

pub use scanner::{MonorepoScanner, ScanResult};
pub use github::GitHubClient;
pub use report::MonorepoReport;
pub use search::{MonorepoSearch, SearchResult};
pub use health::MonorepoHealthScore;
pub use config::IntelligenceConfig;

pub struct MonorepoIntelligence {
    root_path: String,
    config: IntelligenceConfig,
}

impl MonorepoIntelligence {
    pub fn new(root_path: impl Into<String>) -> Self {
        Self {
            root_path: root_path.into(),
            config: IntelligenceConfig::default(),
        }
    }

    pub fn with_config(mut self, config: IntelligenceConfig) -> Self {
        self.config = config;
        self
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
}
