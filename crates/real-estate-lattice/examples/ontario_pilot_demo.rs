//! Ontario Pilot Demo — RREL v14.3
//!
//! Demonstrates the new stabilized v14.3 Real Estate Lattice capabilities:
//! - PropertyTypeClassifier + DealTypeClassifier
//! - FormMappingEngine
//! - OfferPackageAssembler + OfferPackageValidator
//! - MultiOfferTrackEngine
//! - StatusCertificateAnalyzer + DeveloperRiskEngine
//! - OfferRiskSummary helper
//!
//! Run with:
//!   cargo run --example ontario_pilot_demo

use real_estate_lattice::{
    CanadaPilotModule, OfferRiskSummary, OntarioOfferFlowReport,
};
use patsagi_councils::WorldGovernanceEngine;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() {
    println!("🇨🇦 RREL v14.3 Ontario Pilot Demo\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();

    let mut pilot = CanadaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    // ============================================
    // Scenario 1: Standard Resale Deal
    // ============================================
    println!("=== Scenario 1: Resale Single Family Home ===\n");

    let flow1: OntarioOfferFlowReport = pilot
        .process_ontario_offer_flow(
            "LOT 15, PLAN 12345, CITY OF RICHMOND HILL",
            "resale detached home - motivated seller",
            None,
            false,
        )
        .await
        .expect("Failed to process resale flow");

    let summary1 = OfferRiskSummary::from_flow_report(&flow1);
    print_flow_summary("Resale Detached", &flow1, &summary1);

    // ============================================
    // Scenario 2: Pre-Construction Condo with Status Certificate
    // ============================================
    println!("\n=== Scenario 2: Pre-Construction Condo ===\n");

    let flow2: OntarioOfferFlowReport = pilot
        .process_ontario_offer_flow(
            "UNIT 1407, LEVEL 14, TORONTO STANDARD CONDOMINIUM PLAN 56789",
            "new pre-construction from reputable builder",
            Some("Status certificate: Reserve fund healthy, minor special assessment expected"),
            true,
        )
        .await
        .expect("Failed to process pre-construction flow");

    let summary2 = OfferRiskSummary::from_flow_report(&flow2);
    print_flow_summary("Pre-Construction Condo", &flow2, &summary2);

    println!("\n✅ Ontario Pilot Demo complete. All v14.3 modules exercised successfully.");
    println!("   Ready for AlphaProMega Real Estate Inc. use.");
}

fn print_flow_summary(title: &str, flow: &OntarioOfferFlowReport, summary: &OfferRiskSummary) {
    println!("📋 {} ", title);
    println!("   Deal Type           : {}", flow.deal_type);
    println!("   Recommended Form    : {}", flow.recommended_form);
    println!("   Offer Valid         : {}", flow.offer_valid);
    println!("   Escalation Recommended: {}", flow.multi_offer_escalation_triggered);
    if let Some(risk) = &flow.status_certificate_risk {
        println!("   Status Certificate  : {}", risk);
    }
    if let Some(risk) = &flow.developer_risk {
        println!("   Developer Risk      : {}", risk);
    }
    println!("   Overall Mercy       : {:.2}", flow.overall_mercy);
    println!("   Summary             : {}", summary.summary);
    println!();
}
