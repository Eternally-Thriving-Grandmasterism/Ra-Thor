//! # Ra-Thor Monorepo Intelligence v0.2.0
//!
//! Infinite-grade monorepo intelligence system.
//! Designed for universal integration with Grok, Ra-Thor, and any future AI system.

pub mod scanner;
pub mod github;
pub mod report;
pub mod search;
pub mod health;

pub use scanner::{MonorepoScanner, ScanResult};
pub use github::GitHubClient;
pub use report::MonorepoReport;
pub use search::MonorepoSearch;
pub use health::MonorepoHealthScore;

/// The ultimate monorepo intelligence interface.
/// Works seamlessly with any AI system (Grok, Claude, custom agents, etc.).
pub struct MonorepoIntelligence {
    root_path: String,
    config: IntelligenceConfig,
}

#[derive(Debug, Clone)]
pub struct IntelligenceConfig {
    pub include_hidden: bool,
    pub max_depth: Option<usize>,
    pub enable_parallel: bool,
    pub generate_html_reports: bool,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            include_hidden: false,
            max_depth: None,
            enable_parallel: true,
            generate_html_reports: true,
        }
    }
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

    /// Full monorepo scan (parallel when enabled)
    pub async fn scan(&self) -> Result<ScanResult, String> {
        let scanner = scanner::MonorepoScanner::new(&self.root_path)
            .include_hidden(self.config.include_hidden);

        if self.config.enable_parallel {
            // Parallel scanning (future enhancement)
            scanner.scan()
        } else {
            scanner.scan()
        }
    }

    /// Generate a Powrush-focused intelligence report
    pub async fn generate_powrush_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let report = report::MonorepoReport::from_scan(&scan, Some("powrush"));
        Ok(report.to_markdown())
    }

    /// Generate HTML report (beautiful dashboard)
    pub async fn generate_html_report(&self) -> Result<String, String> {
        let scan = self.scan().await?;
        let report = report::MonorepoReport::from_scan(&scan, None);
        Ok(report.to_html())
    }

    /// Smart search with relevance scoring
    pub async fn search(&self, keyword: &str) -> Result<Vec<search::SearchResult>, String> {
        let scan = self.scan().await?;
        let searcher = search::MonorepoSearch::new(scan.files);
        Ok(searcher.search(keyword))
    }

    /// Get health score for Powrush or any module
    pub async fn get_health_score(&self, module: &str) -> Result<health::MonorepoHealthScore, String> {
        let scan = self.scan().await?;
        Ok(health::MonorepoHealthScore::calculate(&scan, module))
    }
}
