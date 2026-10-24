//! ra-thor-legal-lattice — internal legal research notes tool.
//!
//! Not legal advice, not a lawyer or legal team, not a legal product.
//! See `docs/compliance/` (DO-NOT-SHIP, PUBLIC-CLAIM-DISCIPLINE, REFUSAL-20 R07).
//!
//! LL-COMPILE-1: minimal compiling core with a single error type (`LegalError`).
//! Modules that do not yet compile are kept on disk (history preserved) but are
//! not built by default:
//! - `sensor_fusion_bridge` — behind feature `legacy-sensor-bridge` (known broken:
//!   references `CEHICalculator`, `GeneMethylation`, etc. that do not exist, and `chrono`).
//! - `sovereign_legal_frameworks.rs` — not declared (imports missing modules and
//!   `wasm_bindgen`). Left untouched for a later card.

pub mod amendment28;
pub mod cehi;
pub mod fund;
pub mod gates;

#[cfg(feature = "legacy-sensor-bridge")]
pub mod sensor_fusion_bridge;

pub use amendment28::Amendment28Validator;
pub use cehi::FiveGeneCEHI;
pub use fund::MercyLegacyFundEngine;
pub use gates::GatesValidator;

use thiserror::Error;

/// The one error type for this crate.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LegalError {
    #[error("28th Amendment research-concept check failed: {0}")]
    Amendment28Violation(String),

    #[error("7 Mercy-Checks research check failed: {0}")]
    GatesValidationFailed(String),
}
