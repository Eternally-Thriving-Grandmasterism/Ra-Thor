/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// ═════════════════════════════════════════════════════════════════
/// MIGRATED v14.9.3 — Production surface lives at:
///
///   crates/github-connector/
///
/// Prefer:
///
///   use github_connector::{GitHubConnector, CreatePullRequestResponse};
///
/// This root file is retained for historical reference and for any
/// modules that still path-include it. New code should depend on the
/// workspace crate. Missing create_pull_request has been restored in
/// the crate; get_ref_sha + flush_evolution_intents added.
/// ═════════════════════════════════════════════════════════════════

// Re-export the packaged crate when available as a dependency of the
// consuming package. Standalone inclusion of this file still compiles
// the historical surface below for transitional builds.

pub use self::legacy::*;

mod legacy {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/crates/github-connector/src/lib.rs"));
}
