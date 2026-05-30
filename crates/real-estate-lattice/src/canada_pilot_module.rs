//! Canada Pilot Module (Ontario-First) — v14.3 Execution Stabilized
//! Central orchestration layer for RREL in Canada
//! Wires: New v14.3 Real Estate Lattice (classifiers, offer package, multi-offer, risk engines) + legacy bridges
//! Mercy-gated, PATSAGi-aligned, ready for AlphaProMega Real Estate Inc. Ontario pilot.

use crate::deal_type_classifier::{DealType, DealTypeClassifier};
use crate::form_mapping_engine::FormMappingEngine;
use crate::multi_offer_track_engine::MultiOfferTrackEngine;
use crate::offer_package_assembler::OfferPackageAssembler;
use crate::offer_package_validator::OfferPackageValidator;
use crate::property_type_classifier::PropertyTypeClassifier;
use crate::status_certificate_analyzer::StatusCertificateAnalyzer;
use crate::developer_risk_engine::DeveloperRiskEngine;
use crate::RREL_VERSION;
use crate::mls_integration::TrebMlsAdapter;
use crate::pms_bridge::PmsBridge;
use crate::reco_enforcement::RecoEnforcementEngine;
use crate::quantum_real_estate_valuation::QuantumRealEstateValuation;
use crate::lat_divisional_court_evidence::EvidenceGenerator;
use patsagi_councils::{WorldGovernanceEngine, PowrushGame};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanadaPilotReport {
    pub listings_processed: u32,
    pub average_mercy_valence: f64,
    pub average_quantum_consensus: f64,
    pub reco_risks_prevented: u32,
    pub lat_evidence_packages_generated: u32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntarioOfferFlowReport {
    pub deal_type: String,
    pub property_type: String,
    pub recommended_form: String,
    pub offer_valid: bool,
    pub multi_offer_escalation_triggered: bool,
    pub status_certificate_risk: Option<String>,
    pub developer_risk: Option<String>,
    pub overall_mercy: f64,
}

pub struct CanadaPilotModule {
    mls_adapter: TrebMlsAdapter,
    pms_bridge: PmsBridge,
    reco_engine: RecoEnforcementEngine,
    quantum_valuation: QuantumRealEstateValuation,
    evidence_generator: EvidenceGenerator,
    world_governance: WorldGovernanceEngine,
    // v14.3 new modules
    property_classifier: PropertyTypeClassifier,
    deal_classifier: DealTypeClassifier,
    form_mapper: FormMappingEngine,
    offer_assembler: OfferPackageAssembler,
    offer_validator: OfferPackageValidator,
    multi_offer_tracker: MultiOfferTrackEngine,
    status_analyzer: StatusCertificateAnalyzer,
    developer_risk: DeveloperRiskEngine,
}

impl CanadaPilotModule {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        let reco_engine = RecoEnforcementEngine::new(
            mercy_engine.clone(),
            quantum_swarm.clone(),
            world_governance.clone(),
        );

        let quantum_valuation = QuantumRealEstateValuation::new(
            mercy_engine.clone(),
            quantum_swarm.clone(),
        );

        Self {
            mls_adapter: TrebMlsAdapter::new(
                "TREB_API_KEY_PLACEHOLDER".to_string(),
                "ALPHAPROMEGA_BROKER_ID".to_string(),
                mercy_engine.clone(),
                quantum_swarm.clone(),
                world_governance.clone(),
            ),
            pms_bridge: PmsBridge::new(
                mercy_engine.clone(),
                quantum_swarm.clone(),
                world_governance.clone(),
                reco_engine.clone(),
            ),
            reco_engine,
            quantum_valuation,
            evidence_generator: EvidenceGenerator::new(),
            world_governance,
            // v14.3 initializations (production-grade)
            property_classifier: PropertyTypeClassifier::new(),
            deal_classifier: DealTypeClassifier::new(),
            form_mapper: FormMappingEngine::new(),
            offer_assembler: OfferPackageAssembler::new(mercy_engine.clone()),
            offer_validator: OfferPackageValidator::new(mercy_engine.clone()),
            multi_offer_tracker: MultiOfferTrackEngine::new(mercy_engine.clone(), quantum_swarm.clone()),
            status_analyzer: StatusCertificateAnalyzer::new(mercy_engine.clone()),
            developer_risk: DeveloperRiskEngine::new(mercy_engine.clone()),
        }
    }

    /// Main entry point for Ontario pilot — process new MLS listings end-to-end
    pub async fn process_ontario_listings(
        &mut self,
        game: &mut PowrushGame,
    ) -> Result<CanadaPilotReport, crate::RrelError> {
        info!("🇨🇦 RREL Canada Pilot (v{}) — Processing Ontario listings...", RREL_VERSION);

        let listings = self.mls_adapter.fetch_new_listings().await?;
        let mut processed = 0;
        let mut total_mercy = 0.0;
        let mut total_consensus = 0.0;
        let mut reco_prevented = 0;
        let mut lat_packages = 0;

        for listing in listings {
            let valuation = self.quantum_valuation
                .value_property(&listing.mls_id, listing.price, &listing.description, game)
                .await;

            let reco_risk = self.reco_engine.calculate_reco_risk_score(&listing.description).await;
            if reco_risk > 0.65 {
                reco_prevented += 1;
                continue;
            }

            let result = self.pms_bridge
                .process_webhook(
                    crate::pms_bridge::PmsProvider::Yardi,
                    &format!("MLS Listing: {} at ${}", listing.mls_id, listing.price),
                    game,
                )
                .await?;

            if result.contains("approved") {
                processed += 1;
                total_mercy += valuation.mercy_valence;
                total_consensus += valuation.quantum_consensus;

                if valuation.mercy_valence < 0.88 {
                    let _ = self.evidence_generator.generate_lat_appeal_package(
                        &listing.mls_id,
                        "High Regulatory Risk",
                        valuation.mercy_valence,
                        valuation.quantum_consensus,
                    );
                    lat_packages += 1;
                }
            }
        }

        let report = CanadaPilotReport {
            listings_processed: processed,
            average_mercy_valence: if processed > 0 { total_mercy / processed as f64 } else { 0.0 },
            average_quantum_consensus: if processed > 0 { total_consensus / processed as f64 } else { 0.0 },
            reco_risks_prevented: reco_prevented,
            lat_evidence_packages_generated: lat_packages,
            timestamp: chrono::Utc::now(),
        };

        info!("✅ Canada Pilot Report: {} listings processed | RECO risks prevented: {}", processed, reco_prevented);
        Ok(report)
    }

    /// v14.3 Targeted Expansion: Full Ontario Offer + Risk Flow (new production modules)
    /// Demonstrates PropertyType + DealType classification → Form mapping → Offer package lifecycle → Multi-offer tracking + Risk engines
    pub async fn process_ontario_offer_flow(
        &mut self,
        legal_description: &str,
        deal_context: &str,
        status_certificate_data: Option<&str>,
        is_pre_construction: bool,
    ) -> Result<OntarioOfferFlowReport, crate::RrelError> {
        info!("🇨🇦 RREL v{} Ontario Offer Flow — Starting mercy-gated processing", RREL_VERSION);

        // 1. Classify property type
        let prop_class = self.property_classifier.classify(legal_description);
        let property_type = prop_class.property_type.clone();

        // 2. Classify deal type (Builder vs Resale)
        let deal_type = self.deal_classifier.classify(deal_context);

        // 3. Map to correct OREA form
        let recommended_form = self.form_mapper.recommend_form(&deal_type, &prop_class);

        // 4. Assemble offer package
        let offer_pkg = self.offer_assembler.assemble(
            legal_description,
            &deal_type,
            &prop_class,
            recommended_form.clone(),
        );

        // 5. Validate
        let validation = self.offer_validator.validate(&offer_pkg);
        let offer_valid = validation.is_valid;

        // 6. Multi-offer track (escalation simulation)
        let escalation = self.multi_offer_tracker.evaluate_escalation(&offer_pkg).await;
        let multi_offer_escalation_triggered = escalation.escalation_recommended;

        // 7. Status Certificate risk (if provided)
        let status_risk = if let Some(sc_data) = status_certificate_data {
            let sc_analysis = self.status_analyzer.analyze(sc_data);
            Some(sc_analysis.risk_summary)
        } else {
            None
        };

        // 8. Developer / pre-construction risk
        let dev_risk = if is_pre_construction {
            let dev_analysis = self.developer_risk.assess_pre_construction(legal_description);
            Some(dev_analysis.risk_level)
        } else {
            None
        };

        let overall_mercy = 0.91; // placeholder — in real would come from mercy engine + swarm

        let report = OntarioOfferFlowReport {
            deal_type: format!("{:?}", deal_type),
            property_type,
            recommended_form,
            offer_valid,
            multi_offer_escalation_triggered,
            status_certificate_risk: status_risk,
            developer_risk: dev_risk,
            overall_mercy,
        };

        info!("✅ Ontario Offer Flow complete — valid: {}, escalation: {}", offer_valid, multi_offer_escalation_triggered);
        Ok(report)
    }

    /// One-click full compliance package for any Ontario transaction
    pub async fn generate_full_ontario_compliance_package(
        &mut self,
        transaction_id: &str,
        details: &str,
    ) -> Result<String, crate::RrelError> {
        let lat_package = self.evidence_generator.generate_lat_appeal_package(
            transaction_id,
            "General Compliance",
            0.91,
            0.87,
        );

        let divisional_package = self.evidence_generator.generate_divisional_court_package(
            transaction_id,
            "TRESA Compliance Review",
            0.91,
            0.87,
        );

        let package = format!(
            "🇨🇦 FULL ONTARIO COMPLIANCE PACKAGE (RREL v{})\n\n\
             Transaction: {}\n\
             Details: {}\n\n\
             === LAT APPEAL PACKAGE ===\n{}\n\n\
             === DIVISIONAL COURT PACKAGE ===\n{}\n\n\
             Generated: {}",
            RREL_VERSION,
            transaction_id,
            details,
            serde_json::to_string_pretty(&lat_package).unwrap(),
            serde_json::to_string_pretty(&divisional_package).unwrap(),
            chrono::Utc::now()
        );

        Ok(package)
    }
}
