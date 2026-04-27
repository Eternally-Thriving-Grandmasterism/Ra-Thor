//! # Ra-Thor Monorepo Intelligence
//!
//! Universal monorepo intelligence system for Grok, Ra-Thor, and any AI framework.
//! Provides exhaustive scanning, GitHub pagination, smart search, and structured reporting.

pub mod scanner;
pub mod github;
pub mod report;
pub mod search;

pub use scanner::{MonorepoScanner, ScanResult};
pub use github::GitHubClient;
pub use report::MonorepoReport;
pub use search::MonorepoSearch;

/// Main unified interface for monorepo intelligence.
/// Works with Grok, other LLMs, and any AI system.
pub struct MonorepoIntelligence {
    root_path: String,
}

impl MonorepoIntelligence {
    pub fn new(root_path: impl Into<String>) -> Self {
        Self {
            root_path: root_path.into(),
        }
    }

    /// Scan the entire monorepo
    pub fn scan(&self) -> Result<ScanResult, String> {
        let scanner = scanner::MonorepoScanner::new(&self.root_path);
        scanner.scan()
    }

    /// Generate a Powrush-focused report
    pub fn generate_powrush_report(&self) -> Result<String, String> {
        let scan = self.scan()?;
        let report = report::MonorepoReport::from_scan(&scan, Some("powrush"));
        Ok(report.to_markdown())
    }

    /// Search for any keyword with smart relevance scoring
    pub fn search(&self, keyword: &str) -> Result<Vec<search::SearchResult>, String> {
        let scan = self.scan()?;
        let searcher = search::MonorepoSearch::new(scan.files);
        Ok(searcher.search(keyword))
    }

    /// Get GitHub client (for external repo scanning)
    pub fn github_client(&self, token: Option<String>) -> GitHubClient {
        GitHubClient::new(token)
    }
}
