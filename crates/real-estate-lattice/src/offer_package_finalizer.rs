//! Offer Package Finalizer

use crate::offer_package_assembler::OfferPackage;
use crate::disclosure_manager::DisclosureState;

#[derive(Debug, Clone)]
pub struct FinalizedPackage {
    pub offer_package: OfferPackage,
    pub disclosure_summary: String,
    pub patsagi_alignment_score: f32,
    pub ready_for_submission: bool,
}

pub struct OfferPackageFinalizer;

impl OfferPackageFinalizer {
    pub fn finalize(offer_package: OfferPackage, disclosure_state: &DisclosureState) -> FinalizedPackage {
        let score = if offer_package.validation.is_valid { 0.9 } else { 0.75 };
        FinalizedPackage {
            offer_package,
            disclosure_summary: crate::disclosure_manager::DisclosureManager::generate_disclosure_summary(disclosure_state),
            patsagi_alignment_score: score,
            ready_for_submission: offer_package.validation.is_valid
        }
    }
}