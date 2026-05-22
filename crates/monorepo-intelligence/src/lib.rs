pub mod config;
pub mod github;
pub mod health;
pub mod inheritance; // NEW
pub mod plugin;
pub mod report;
pub mod scanner;
pub mod search;

pub use inheritance::{analyze_inheritance, InheritanceStatus};

// Re-exports from other modules
pub use scanner::MonorepoScanner;
pub use health::MonorepoHealthScore;
pub use search::{MonorepoSearch, SearchResult};
pub use report::MonorepoReport;
pub use plugin::{MonorepoPlugin, PluginResult};

use std::path::PathBuf;

#[derive(Debug)]
pub struct MonorepoIntelligence {
    pub root_path: PathBuf,
    // ... other fields
}

impl MonorepoIntelligence {
    pub fn new(root: PathBuf) -> Self {
        Self { root_path: root }
    }

    // TODO: Integrate inheritance analysis into main workflow
}
