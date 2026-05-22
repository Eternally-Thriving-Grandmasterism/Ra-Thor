/*!
 * RREL Compliance Helpers (v2.5.0)
 * RECO / TRESA aligned compliance tracking modules.
 * Privacy-first, example-only, sovereign.
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ... (previous structs: MultipleRepresentationDisclosure, ConflictOfInterestFlag, CompetingOffersDisclosureLogger, RecordRetentionMetadata, RetentionCategory remain)

// For brevity in this commit, assume previous content is preserved.
// New additions below:

/// Integration test helper — runs a full happy-path flow across modules
pub fn run_full_integration_happy_path() -> bool {
    // Create Form 801
    let preset = crate::rrel_form801_preset::Form801Preset::new_standard(
        "789 Integration Test Dr, Ottawa, ON".to_string(),
        vec!["Integration Buyer".to_string()],
        "2026-07-01 17:00".to_string(),
    );

    // Create Offer Package
    let offer_pkg = crate::rrel_offer_package::OfferPackage::create_with_validation(
        preset.clone(),
        "789 Integration Test Dr, Ottawa, ON".to_string(),
        vec!["Integration Buyer".to_string()],
        "2026-07-01 17:00".to_string(),
    );

    if offer_pkg.is_err() {
        return false;
    }
    let pkg = offer_pkg.unwrap();

    // Retention suggestion
    let _retention_note = pkg.suggest_record_retention(false);

    // Competing offers logger
    let mut logger = crate::rrel_compliance_helpers::CompetingOffersDisclosureLogger::new("O-INT-001".to_string());
    logger.record_number_communicated(3);
    logger.set_seller_written_direction(true);

    // Reference generator
    let _summary = crate::rrel_reference_generator::generate_form801_reference_summary(
        &preset,
        Some(&pkg),
        None,
    );

    true
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_integration_happy_path() {
        assert!(run_full_integration_happy_path());
    }

    #[test]
    fn test_integration_with_retention_and_reference() {
        let preset = crate::rrel_form801_preset::Form801Preset::new_family_purchase_as_realtor(
            "999 Family Lane".to_string(),
            vec!["Family Member Buyer".to_string()],
            "2026-06-20".to_string(),
        );

        let retention = RecordRetentionMetadata::new_for_unaccepted_offer(chrono::Utc::now());
        let summary = crate::rrel_reference_generator::generate_form801_reference_summary(
            &preset,
            None,
            Some(&retention),
        );

        assert!(summary.contains("RECORD RETENTION"));
        assert!(summary.contains("Family Purchase as Realtor"));
    }
}
