//! USA MLS Adapter — RREL v0.6.0
//! Generic + State-Specific MLS Integration Layer for All 50 States
//!
//! Derived from RREL-USA-Expansion-Codex-v0.6.0.md
//!
//! RREL-UNGATE-1: ungated — powrush/patsagi/mercy/quantum-swarm imports replaced
//! with local std-only stubs. process_usa_listing removed (depended on external
//! regulatory engine). fetch_new_listings NotConfigured behaviour preserved (#730).

use crate::RREL_VERSION;
use crate::rrel_compliance::AiImageRecord;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum UsaMlsError {
    #[error("MLS fetch failed for state {state}: {message}")]
    MlsFetchFailed { state: String, message: String },
    #[error("No real MLS connection configured for state {0}")]
    NotConfigured(String),
}

// ---------------------------------------------------------------------------
// UsaListing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaListing {
    pub mls_id: String,
    pub state: String,
    /// Price as reported by the MLS data source. None when not yet sourced.
    pub price: Option<f64>,
    pub description: String,
    /// Images with alteration labels per RECO AI guidance.
    pub photos: Vec<AiImageRecord>,
    pub address: String,
    /// Identifier of the verified data source (e.g. MLS feed name + record ID).
    pub source: Option<String>,
    /// True only after a registrant has cross-checked against verifiable property documents.
    pub registrant_verified: bool,
}

// ---------------------------------------------------------------------------
// UsaMlsAdapter
// ---------------------------------------------------------------------------

/// MLS adapter stub. No live MLS connection is wired; all state-specific
/// fetch paths return `NotConfigured`. Wire a real MLS feed to replace.
pub struct UsaMlsAdapter;

impl UsaMlsAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Fetch new listings for any USA state.
    ///
    /// Returns `Err(NotConfigured)` when no real MLS connection is available.
    /// A real implementation would call state-specific MLS APIs (CRMLS, FMLS, HAR,
    /// OneKey, etc.) with proper auth and return listings with verified sources.
    ///
    /// No price is emitted until a sourced, registrant-verified listing is available.
    pub async fn fetch_new_listings(&self, state: &str) -> Result<Vec<UsaListing>, UsaMlsError> {
        info!(
            "🇺🇸 Fetching new MLS listings for {} (RREL v{}) — no live MLS connection",
            state, RREL_VERSION
        );

        // No real MLS connection is wired. Return NotConfigured so callers know
        // that no data (not invented data) is available. Wire a real MLS feed here.
        Err(UsaMlsError::NotConfigured(state.to_string()))
    }
}

impl Default for UsaMlsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test-only mock data — clearly scoped to #[cfg(test)]
// ---------------------------------------------------------------------------

#[cfg(test)]
pub mod test_fixtures {
    use super::*;

    /// Returns a clearly-labelled mock listing for unit tests only.
    /// This data is synthetic and must never be presented as real property information.
    pub fn mock_listing(state: &str) -> UsaListing {
        UsaListing {
            mls_id: format!("{}-MOCK-001", state),
            state: state.to_string(),
            price: None, // price requires a cited source
            description: "[MOCK — synthetic test data, not a real property]".to_string(),
            photos: vec![],
            address: "[MOCK — no real address]".to_string(),
            source: Some("test-fixture".to_string()),
            registrant_verified: false,
        }
    }
}
