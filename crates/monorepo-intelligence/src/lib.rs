//! # Ra-Thor Monorepo Intelligence
//!
//! Exhaustive monorepo scanning, GitHub pagination, smart search, and structured reporting.
//! This crate gives Grok/Ra-Thor systems the ability to deeply understand the entire monorepo.

pub mod scanner;
pub mod github;
pub mod report;
pub mod search;

pub use scanner::MonorepoScanner;
pub use github::GitHubClient;
pub use report::MonorepoReport;
pub use search::MonorepoSearch;
