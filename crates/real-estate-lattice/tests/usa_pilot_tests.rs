//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! Integration Tests for USA Pilot Module (v14.3)
//! RREL-BUILD-1: rewritten to use always-on modules only (UsaPilotModule
//! requires patsagi_councils/powrush and is gated design-only).

use real_estate_lattice::{
    property_type_classifier::{PropertyTypeClassifier, OntarioPropertyType},
    deal_type_classifier::{DealTypeClassifier, DealType},
    developer_risk_engine::DeveloperRiskEngine,
    status_certificate_analyzer::StatusCertificateAnalyzer,
    lat_divisional_court_evidence::EvidenceGenerator,
    deal_readiness_scoring::calculate_deal_readiness_score,
    RREL_VERSION,
};

#[test]
fn test_usa_offer_flow_basic_california() {
    let property = PropertyTypeClassifier::classify_from_legal_description(
        "LOT 15, TRACT 12345, CITY OF LOS ANGELES, STATE OF CALIFORNIA",
    );
    let deal = DealTypeClassifier::classify(
        "Standard home purchase in California with full disclosures",
        &property.property_type,
    );
    assert!(property.confidence >= 0.0 && property.confidence <= 1.0);
    println!(
        "usa_basic_california: property={:?} deal={:?}",
        property.property_type, deal.deal_type
    );
}

#[test]
fn test_usa_offer_flow_florida_condo() {
    let property = PropertyTypeClassifier::classify_from_legal_description(
        "UNIT 801, OCEAN VIEW CONDOMINIUMS, MIAMI-DADE COUNTY, FLORIDA",
    );
    let deal = DealTypeClassifier::classify(
        "Condo purchase in Miami with HOA review and status certificate",
        &property.property_type,
    );
    let cert = StatusCertificateAnalyzer::analyze("HOA reserve fund at 65% funded");
    assert!(!cert.overall_risk_level.is_empty());
    println!(
        "usa_florida_condo: property={:?} deal={:?} cert_risk={}",
        property.property_type, deal.deal_type, cert.overall_risk_level
    );
}

#[test]
fn test_usa_offer_flow_high_value_texas() {
    let property = PropertyTypeClassifier::classify_from_legal_description(
        "LOT 5, BLOCK 3, HIGHLAND PARK ESTATES, DALLAS COUNTY, TEXAS",
    );
    let deal = DealTypeClassifier::classify(
        "Luxury single family home purchase in Highland Park, Texas",
        &property.property_type,
    );
    let score = calculate_deal_readiness_score(90, 0.88, "texas", false, 0.95);
    println!(
        "usa_high_value_texas: property={:?} deal={:?} readiness={}",
        property.property_type, deal.deal_type, score
    );
}

#[test]
fn test_usa_offer_flow_with_kickback_new_york() {
    let property = PropertyTypeClassifier::classify_from_legal_description(
        "UNIT 4B, 123 PARK AVENUE COOPERATIVE, MANHATTAN, NEW YORK",
    );
    let dev_risk = DeveloperRiskEngine::assess(
        "Manhattan Cooperative Board",
        "Co-op board approval required; previous application denial rate 40%",
    );
    assert!(dev_risk.overall_risk_score >= 0.0);
    println!(
        "usa_new_york_kickback: property={:?} risk={:.2}",
        property.property_type, dev_risk.overall_risk_score
    );
}

#[test]
fn test_usa_developer_risk_engine() {
    let risk = DeveloperRiskEngine::assess(
        "US National Builder Corp",
        "Multiple active litigations in Florida and Texas",
    );
    assert!(risk.overall_risk_score >= 0.0 && risk.overall_risk_score <= 1.0);
    println!("usa_developer_risk: {:.3}", risk.overall_risk_score);
}

#[test]
fn test_usa_lat_evidence_generator() {
    let gen = EvidenceGenerator::new();
    let package = gen.generate_lat_appeal_package(
        "USA-TEST-001",
        "USA pilot evidence package test",
        0.88,
        0.82,
    );
    assert_eq!(package.case_id, "USA-TEST-001");
    println!("usa_lat_evidence: case={}", package.case_id);
}

#[test]
fn test_usa_version_and_scoring() {
    assert!(RREL_VERSION.starts_with("14."), "RREL_VERSION must be 14.x: {}", RREL_VERSION);
    let score = calculate_deal_readiness_score(85, 0.78, "california", false, 0.90);
    println!("usa_version_and_scoring: version={} score={}", RREL_VERSION, score);
    let prop = PropertyTypeClassifier::classify_from_legal_description(
        "LOT 1, BLOCK A, ANYWHERE USA",
    );
    assert_ne!(prop.property_type, OntarioPropertyType::Condominium);
    println!("property_type={:?}", prop.property_type);
}
