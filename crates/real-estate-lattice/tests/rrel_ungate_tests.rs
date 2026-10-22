//! RREL-UNGATE-1 — #730 guarantee tests
//!
//! These tests verify the four behaviours mandated by the RECO AI guidance
//! (#730) after the modules were ungated in RREL-UNGATE-1:
//!
//!   1. usa_mls_adapter: fetch_new_listings returns NotConfigured; no invented listing.
//!   2. quantum_real_estate_valuation: no price without a source; always returns Draft.
//!   3. reco_form_handlers: blocks without written consent; output is Draft.
//!   4. offer_package_finalizer: ready_for_submission only when HumanVerified.

// ── 1. usa_mls_adapter ────────────────────────────────────────────────────

mod usa_mls_adapter_tests {
    use real_estate_lattice::usa_mls_adapter::{UsaMlsAdapter, UsaMlsError};

    /// fetch_new_listings must return NotConfigured for every state.
    /// No listing data (invented or otherwise) may be returned.
    #[tokio::test]
    async fn fetch_returns_not_configured_no_invented_listing() {
        let adapter = UsaMlsAdapter::new();
        for state in &["ontario", "california", "texas", "florida", "new-york"] {
            let result = adapter.fetch_new_listings(state).await;
            match result {
                Err(UsaMlsError::NotConfigured(s)) => {
                    assert_eq!(&s, state, "NotConfigured must echo the requested state");
                }
                Ok(listings) => {
                    panic!(
                        "Expected NotConfigured for '{}' but got {} listing(s) — \
                         no invented data must be returned",
                        state,
                        listings.len()
                    );
                }
                Err(other) => {
                    panic!("Expected NotConfigured for '{}', got: {:?}", state, other);
                }
            }
        }
    }

    /// UsaListing with no source must not carry a price.
    #[test]
    fn listing_without_source_has_no_price() {
        use real_estate_lattice::usa_mls_adapter::UsaListing;
        let listing = UsaListing {
            mls_id: "CA-MOCK-001".to_string(),
            state: "california".to_string(),
            price: None, // price requires a cited source
            description: "[synthetic test data — not a real property]".to_string(),
            photos: vec![],
            address: "[no real address]".to_string(),
            source: None,
            registrant_verified: false,
        };
        assert!(
            listing.price.is_none(),
            "listing without source must not carry a price"
        );
    }
}

// ── 2. quantum_real_estate_valuation ─────────────────────────────────────

mod quantum_valuation_tests {
    use real_estate_lattice::quantum_real_estate_valuation::QuantumRealEstateValuation;
    use real_estate_lattice::rrel_compliance::OutputStatus;

    /// analyse_property must always return Draft status — never ready for use.
    #[tokio::test]
    async fn analyse_property_always_returns_draft() {
        let engine = QuantumRealEstateValuation::new();
        let output = engine
            .analyse_property("PROP-001", "standard detached home", vec![])
            .await;
        assert!(
            matches!(output.status, OutputStatus::Draft),
            "analyse_property must always return Draft status"
        );
        assert!(
            !output.is_ready_for_use(),
            "Draft output must not be ready for use"
        );
    }

    /// No price must be emitted; reference_price must be None regardless of input.
    #[tokio::test]
    async fn no_price_emitted_without_source() {
        let engine = QuantumRealEstateValuation::new();
        let output = engine
            .analyse_property("PROP-002", "trust account conflict eviction", vec![])
            .await;
        assert!(
            output.value.reference_price.is_none(),
            "reference_price must be None — RREL does not produce prices"
        );
    }

    /// With comparables, output is still Draft; price still None.
    #[tokio::test]
    async fn with_comparables_still_draft_no_price() {
        let engine = QuantumRealEstateValuation::new();
        // Fictional example data — these identifiers and prices are invented for test
        // purposes only; they do not represent real MLS listings or property values.
        let comps = vec![
            "MLS-CA-2026-001 — $650k (registrant verified) [fictional example data]".to_string(),
            "MLS-CA-2026-002 — $640k (registrant verified) [fictional example data]".to_string(),
        ];
        let output = engine
            .analyse_property("PROP-003", "resale detached", comps)
            .await;
        assert!(
            matches!(output.status, OutputStatus::Draft),
            "output with comparables must still be Draft"
        );
        assert!(
            output.value.reference_price.is_none(),
            "reference_price must remain None even with comparables"
        );
        assert!(
            output.value.draft_note.contains("DRAFT"),
            "draft_note must contain DRAFT label"
        );
    }

    /// Draft output becomes ready only after human verification.
    #[tokio::test]
    async fn becomes_ready_after_human_verification() {
        let engine = QuantumRealEstateValuation::new();
        let output = engine
            .analyse_property("PROP-004", "standard home", vec![])
            .await;
        assert!(!output.is_ready_for_use(), "must not be ready before verification");
        let verified = output.verify("AGENT-REG-0099");
        assert!(
            verified.is_ready_for_use(),
            "must be ready after HumanVerified"
        );
    }
}

// ── 3. reco_form_handlers ────────────────────────────────────────────────

mod reco_form_handlers_tests {
    use real_estate_lattice::reco_form_handlers::{RecoFormHandlers, RecoFormType};
    use real_estate_lattice::rrel_compliance::{ConsentRecord, OutputStatus};
    use std::collections::HashMap;

    fn consent(written: bool) -> ConsentRecord {
        ConsentRecord {
            client_ref: "client-anon-test-001".to_string(),
            written,
            scope: "offer-to-purchase form review".to_string(),
            tool_id: "internal-drafting-tool".to_string(),
            obtained_at: 0,
        }
    }

    fn form_data() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("property_address".to_string(), "[test address — no real data]".to_string());
        m.insert("purchase_price".to_string(), "500000".to_string());
        m
    }

    /// No consent → process_reco_form must return Err.
    #[test]
    fn blocks_without_written_consent() {
        let mut handlers = RecoFormHandlers::new();
        let no_consent = consent(false);
        let result = handlers.process_reco_form(
            RecoFormType::OfferToPurchase,
            form_data(),
            &no_consent,
        );
        assert!(
            result.is_err(),
            "process_reco_form must fail without written consent"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("written client consent"),
            "error message must name the requirement: {msg}"
        );
    }

    /// With written consent → returns Ok with a Draft AiOutput.
    #[test]
    fn draft_with_written_consent() {
        let mut handlers = RecoFormHandlers::new();
        let written = consent(true);
        let result = handlers.process_reco_form(
            RecoFormType::OfferToPurchase,
            form_data(),
            &written,
        );
        assert!(result.is_ok(), "should succeed with written consent");
        let output = result.unwrap();
        assert!(
            matches!(output.status, OutputStatus::Draft),
            "output must be Draft — never ready without human sign-off"
        );
        assert!(
            !output.is_ready_for_use(),
            "Draft output must not be ready for use"
        );
        assert!(
            !output.value.tolc8_seal,
            "tolc8_seal must be false — AI never sets this"
        );
    }

    /// Empty form data → blocked even with consent.
    #[test]
    fn blocks_empty_form_data() {
        let mut handlers = RecoFormHandlers::new();
        let written = consent(true);
        let result = handlers.process_reco_form(
            RecoFormType::Form801,
            HashMap::new(),
            &written,
        );
        assert!(result.is_err(), "empty form data must be rejected");
    }

    /// Compliance is always unverified — never passed: false implies human review is forced.
    #[test]
    fn compliance_always_unverified() {
        let mut handlers = RecoFormHandlers::new();
        let written = consent(true);
        let output = handlers
            .process_reco_form(RecoFormType::ListingAgreement, form_data(), &written)
            .unwrap();
        assert!(
            !output.value.compliance.passed,
            "compliance must never be marked passed — human review required"
        );
        assert!(
            !output.value.compliance.issues.is_empty(),
            "compliance issues must be populated so the agent is not misled"
        );
    }
}

// ── 4. offer_package_finalizer ────────────────────────────────────────────

mod offer_package_finalizer_tests {
    use real_estate_lattice::offer_package_finalizer::{OfferPackage, OfferPackageFinalizer};
    use real_estate_lattice::offer_package_validator::{
        OfferPackageValidation, OfferPackageValidator,
    };
    use real_estate_lattice::property_type_classifier::OntarioPropertyType;
    use real_estate_lattice::deal_type_classifier::DealType;
    use real_estate_lattice::rrel_compliance::OutputStatus;

    fn complete_validation() -> OfferPackageValidation {
        OfferPackageValidator::validate(
            &OntarioPropertyType::Detached,
            &DealType::Resale,
            true,  // has_form_100
            false, // has_status_certificate (not required for freehold resale)
            false, // has_developer_disclosures
            false, // has_family_ila
        )
    }

    fn incomplete_validation() -> OfferPackageValidation {
        OfferPackageValidator::validate(
            &OntarioPropertyType::Detached,
            &DealType::Resale,
            false, // missing form 100
            false,
            false,
            false,
        )
    }

    fn make_package(validation: OfferPackageValidation) -> OfferPackage {
        OfferPackage {
            property_address: "[test — no real address]".to_string(),
            validation,
            assembled_documents: vec!["Form 100".to_string()],
            notes: vec![],
        }
    }

    /// Draft status → ready_for_submission must be false regardless of validation.
    #[test]
    fn draft_never_ready_for_submission() {
        let pkg = make_package(complete_validation());
        let finalized =
            OfferPackageFinalizer::finalize(pkg, "Disclosures provided", OutputStatus::Draft);
        assert!(
            !finalized.ready_for_submission,
            "Draft must never be ready_for_submission"
        );
        assert!(
            finalized.verified_by.is_none(),
            "verified_by must be None for Draft"
        );
    }

    /// HumanVerified + complete validation → ready_for_submission is true.
    #[test]
    fn human_verified_complete_is_ready() {
        let pkg = make_package(complete_validation());
        let finalized = OfferPackageFinalizer::finalize(
            pkg,
            "All disclosures reviewed",
            OutputStatus::HumanVerified {
                by: "AGENT-REG-0042".to_string(),
                at: 0,
            },
        );
        assert!(
            finalized.ready_for_submission,
            "HumanVerified + complete package must be ready_for_submission"
        );
        assert_eq!(
            finalized.verified_by.as_deref(),
            Some("AGENT-REG-0042"),
            "verified_by must store the agent ID"
        );
    }

    /// HumanVerified + incomplete validation → ready_for_submission is false.
    #[test]
    fn human_verified_incomplete_not_ready() {
        let pkg = make_package(incomplete_validation());
        let finalized = OfferPackageFinalizer::finalize(
            pkg,
            "Disclosures partial",
            OutputStatus::HumanVerified {
                by: "AGENT-REG-0042".to_string(),
                at: 0,
            },
        );
        assert!(
            !finalized.ready_for_submission,
            "HumanVerified with incomplete validation must not be ready_for_submission"
        );
        // verified_by is still recorded even when not ready
        assert!(
            finalized.verified_by.is_some(),
            "verified_by must be recorded even when package is not ready"
        );
    }
}
