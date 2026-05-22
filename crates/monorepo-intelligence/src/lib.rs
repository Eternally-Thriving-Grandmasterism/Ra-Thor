pub mod config;
pub mod github;
pub mod health;
pub mod inheritance;
pub mod plugin;
pub mod report;
pub mod scanner;
pub mod search;

pub use inheritance::{analyze_inheritance, InheritanceStatus};
pub use scanner::{MonorepoScanner, ScanResult, ScannedFile, ScanError};

use std::path::PathBuf;

/// Main entry point for Ra-Thor Monorepo Intelligence.
/// Provides high-level access to scanning, inheritance analysis, health scoring, and reporting.
pub struct MonorepoIntelligence {
    pub root_path: PathBuf,
}

impl MonorepoIntelligence {
    pub fn new(root: impl AsRef<std::path::Path>) -> Self {
        Self {
            root_path: root.as_ref().to_path_buf(),
        }
    }

    /// Returns a scanner configured for this monorepo root.
    pub fn scanner(&self) -> MonorepoScanner {
        MonorepoScanner::new(self.root_path.clone())
    }

    /// Performs inheritance compliance analysis across all crates.
    pub fn analyze_inheritance(&self) -> Vec<InheritanceStatus> {
        analyze_inheritance(&self.root_path)
    }

    /// Runs a full scan and returns rich results.
    pub fn full_scan(&self) -> Result<ScanResult, ScanError> {
        self.scanner().scan()
    }
}
