//! github_connector.rs — MIGRATION SHIM (v14.9.3)
//!
//! Production surface has moved to the workspace crate:
//!
//!   crates/github-connector/  (package name: `github-connector`)
//!
//! Prefer in Cargo.toml:
//!
//!   github-connector = { path = "crates/github-connector", version = "14.9.3" }
//!
//! And in code:
//!
//!   use github_connector::{GitHubConnector, CreatePullRequestResponse, GitHubError};
//!
//! Restored / added in the crate vs this historical root file:
//! - create_pull_request (was missing)
//! - get_ref_sha
//! - flush_evolution_intents (drains offline GitHubSurface queues)
//! - base64 0.22 Engine API
//!
//! This file intentionally has no runtime implementation so it cannot
//! diverge from the crate. Historical content lives in git history.
//!
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License

// Compile-time pointer for grep / docs only.
pub const MIGRATED_TO: &str = "crates/github-connector";
pub const CRATE_VERSION: &str = "14.9.3";
