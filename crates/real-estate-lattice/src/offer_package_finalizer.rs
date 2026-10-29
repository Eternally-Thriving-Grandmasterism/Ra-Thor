//! Offer Package Finalizer
//!
//! RECO AI guidance: an offer package must not be submitted unless a registered
//! agent has explicitly verified it. This finalizer refuses to produce a ready-
//! for-submission package when status is still Draft.
//!
//! RREL-UNGATE-1: ungated — offer_package_assembler and disclosure_manager
//! (design-only, no source file) replaced with local minimal types.
//! ValidationResult/OfferPackageValidation mismatch fixed: uses
//! OfferPackageValidation.is_complete (the field that exists) rather than
//! the non-existent is_valid. #730 ready_for_submission behaviour preserved.

use crate::offer_package_validator::OfferPackageValidation;
use crate::rrel_compliance::OutputStatus;

// ---------------------------------------------------------------------------
// OfferPackage — minimal local type; not the full assembler output
// ---------------------------------------------------------------------------

/// Minimal offer package passed into the finalizer. Callers build this from
/// whatever validated source they have (e.g. OfferPackageAssembler output once
/// that module is ungated, or a manually constructed struct in tests).
#[derive(Debug, Clone)]
pub struct OfferPackage {
    pub property_address: String,
    /// Validation produced by OfferPackageValidator.
    pub validation: OfferPackageValidation,
    pub assembled_documents: Vec<String>,
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// FinalizedPackage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FinalizedPackage {
    pub offer_package: OfferPackage,
    /// Free-text disclosure summary provided by the caller.
    pub disclosure_summary: String,
    pub patsagi_alignment_score: f32,
    /// True only when the package has been human-verified by a registered agent.
    pub ready_for_submission: bool,
    /// ID of the agent who verified the package. None until verified.
    pub verified_by: Option<String>,
}

// ---------------------------------------------------------------------------
// OfferPackageFinalizer
// ---------------------------------------------------------------------------

pub struct OfferPackageFinalizer;

impl OfferPackageFinalizer {
    /// Finalize an offer package.
    ///
    /// `agent_verification` must be `OutputStatus::HumanVerified` for the package
    /// to be marked `ready_for_submission`. A Draft status will produce a package
    /// with `ready_for_submission: false` regardless of validation state.
    ///
    /// `disclosure_summary` is a caller-supplied plain-text disclosure statement;
    /// the disclosure_manager module is still design-only.
    pub fn finalize(
        offer_package: OfferPackage,
        disclosure_summary: impl Into<String>,
        agent_verification: OutputStatus,
    ) -> FinalizedPackage {
        let score = if offer_package.validation.is_complete { 0.9 } else { 0.75 };

        let (ready, verified_by) = match agent_verification {
            OutputStatus::HumanVerified { ref by, .. } => {
                // Both validation completeness and human sign-off are required
                (offer_package.validation.is_complete, Some(by.clone()))
            }
            OutputStatus::Draft => {
                // AI draft — must not be submitted without registrant review
                (false, None)
            }
        };

        FinalizedPackage {
            offer_package,
            disclosure_summary: disclosure_summary.into(),
            patsagi_alignment_score: score,
            ready_for_submission: ready,
            verified_by,
        }
    }
}
