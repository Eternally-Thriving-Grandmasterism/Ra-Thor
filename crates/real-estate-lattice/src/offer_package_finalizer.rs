//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! Offer Package Finalizer
//!
//! RECO AI guidance: an offer package must not be submitted unless a registered
//! agent has explicitly verified it. This finalizer refuses to produce a ready-
//! for-submission package when status is still Draft.

use crate::offer_package_assembler::OfferPackage;
use crate::disclosure_manager::DisclosureState;
use crate::rrel_compliance::OutputStatus;

#[derive(Debug, Clone)]
pub struct FinalizedPackage {
    pub offer_package: OfferPackage,
    pub disclosure_summary: String,
    pub patsagi_alignment_score: f32,
    /// True only when the package has been human-verified by a registered agent.
    pub ready_for_submission: bool,
    /// ID of the agent who verified the package. None until verified.
    pub verified_by: Option<String>,
}

pub struct OfferPackageFinalizer;

impl OfferPackageFinalizer {
    /// Finalize an offer package.
    ///
    /// `agent_verification` must be `OutputStatus::HumanVerified` for the package
    /// to be marked `ready_for_submission`. A Draft status will produce a package
    /// with `ready_for_submission: false` regardless of validation state.
    pub fn finalize(
        offer_package: OfferPackage,
        disclosure_state: &DisclosureState,
        agent_verification: OutputStatus,
    ) -> FinalizedPackage {
        let score = if offer_package.validation.is_valid { 0.9 } else { 0.75 };

        let (ready, verified_by) = match agent_verification {
            OutputStatus::HumanVerified { ref by, .. } => {
                // Both validation and human sign-off are required
                (offer_package.validation.is_valid, Some(by.clone()))
            }
            OutputStatus::Draft => {
                // AI draft — must not be submitted without registrant review
                (false, None)
            }
        };

        FinalizedPackage {
            offer_package,
            disclosure_summary: crate::disclosure_manager::DisclosureManager::generate_disclosure_summary(disclosure_state),
            patsagi_alignment_score: score,
            ready_for_submission: ready,
            verified_by,
        }
    }
}
