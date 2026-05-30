//! Lawyer Due Diligence Generator for Real Estate Lattice
//!
//! Produces tailored, production-grade due diligence checklists by combining
//! PropertyType, DealType, StatusCertificateAnalysis, and DeveloperRiskProfile.
//!
//! **Ontario Focus**:
//! - Lawyer-grade checklists aligned with OREA, Condominium Act, Tarion, and RESA
//! - Family transaction sensitivity and ILA prompts
//! - Pre-construction vs resale distinctions
//!
//! **Mercy & Integration**:
//! - Generates clear, prioritized checklists instead of overwhelming lists
//! - Includes PATSAGi / ethical flags where relevant
//! - Designed for direct handoff to lawyer_report_generator and client communication
//!
//! Consumes outputs from PropertyTypeClassifier, DealTypeClassifier,
//! StatusCertificateAnalyzer, and DeveloperRiskEngine.

use crate::property_type_classifier::OntarioPropertyType;
use crate::deal_type_classifier::DealType;
use crate::status_certificate_analyzer::StatusCertificateAnalysis;
use crate::developer_risk_engine::DeveloperRiskProfile;

#[derive(Debug, Clone)]
pub struct DueDiligenceChecklist {
    pub priority_items: Vec<String>,
    pub standard_items: Vec<String>,
    pub ethical_flags: Vec<String>,
    pub recommended_actions: Vec<String>,
}

pub struct LawyerDueDiligenceGenerator;

impl LawyerDueDiligenceGenerator {
    pub fn generate(
        property_type: &OntarioPropertyType,
        deal_type: &DealType,
        status_analysis: Option<&StatusCertificateAnalysis>,
        developer_risk: Option<&DeveloperRiskProfile>,
    ) -> DueDiligenceChecklist {
        let mut priority_items = vec![];
        let mut standard_items = vec![];
        let mut ethical_flags = vec![];
        let mut recommended_actions = vec![];

        // Property type driven
        match property_type {
            OntarioPropertyType::Condominium | OntarioPropertyType::Townhouse => {
                priority_items.push("Review current Status Certificate for special assessments, reserve fund, and litigation".to_string());
                priority_items.push("Confirm condo corporation rules, pet/parking restrictions, and recent AGM minutes".to_string());
            }
            OntarioPropertyType::Detached | OntarioPropertyType::SemiDetached => {
                priority_items.push("Title search, survey / real property report, and zoning compliance".to_string());
                standard_items.push("Environmental concerns, well/septic (if applicable), and encroachment checks".to_string());
            }
            _ => {}
        }

        // Deal type driven
        match deal_type {
            DealType::PreConstruction => {
                priority_items.push("Verify Tarion enrolment and builder warranty details".to_string());
                priority_items.push("Review deposit protection trust account and interim occupancy terms".to_string());
                if let Some(risk) = developer_risk {
                    if risk.overall_risk_score > 0.5 {
                        priority_items.push("Independent legal opinion on builder financials and completion risk".to_string());
                    }
                }
            }
            DealType::FamilyTransfer => {
                ethical_flags.push("Family Transfer detected - confirm Independent Legal Advice (ILA) obtained for all parties".to_string());
                ethical_flags.push("Assess for undue influence or power imbalance. Consider PATSAGi ethical review".to_string());
            }
            DealType::Assignment => {
                priority_items.push("Confirm original APS assignment consent and fee treatment".to_string());
            }
            _ => {}
        }

        if let Some(status) = status_analysis {
            if status.special_assessments_pending || status.litigation_risk {
                priority_items.push("High-risk Status Certificate findings - prepare client for potential cost or timeline impact".to_string());
            }
        }

        standard_items.push("Review all APS conditions, waivers, and deposit terms".to_string());
        standard_items.push("Confirm identification and signing authority".to_string());

        recommended_actions.push("Schedule client review meeting with clear prioritized checklist".to_string());
        recommended_actions.push("Prepare merciful, plain-language summary for clients".to_string());

        DueDiligenceChecklist {
            priority_items,
            standard_items,
            ethical_flags,
            recommended_actions,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::property_type_classifier::OntarioPropertyType;
    use crate::deal_type_classifier::DealType;
    use crate::status_certificate_analyzer::{StatusCertificateAnalyzer, StatusCertificateAnalysis};
    use crate::developer_risk_engine::DeveloperRiskEngine;

    #[test]
    fn integration_condo_resale_with_risky_status_certificate() {
        let status = StatusCertificateAnalyzer::analyze("Special assessment pending. Reserve fund is low.");

        let checklist = LawyerDueDiligenceGenerator::generate(
            &OntarioPropertyType::Condominium,
            &DealType::Resale,
            Some(&status),
            None,
        );

        assert!(checklist.priority_items.iter().any(|i| i.contains("Status Certificate")));
        assert!(checklist.priority_items.iter().any(|i| i.contains("High-risk")));
    }

    #[test]
    fn integration_family_transfer_triggers_ethical_flags() {
        let checklist = LawyerDueDiligenceGenerator::generate(
            &OntarioPropertyType::Detached,
            &DealType::FamilyTransfer,
            None,
            None,
        );

        assert!(checklist.ethical_flags.iter().any(|f| f.contains("Family Transfer")));
        assert!(checklist.ethical_flags.iter().any(|f| f.contains("PATSAGi")));
    }

    #[test]
    fn integration_pre_construction_high_risk_developer() {
        let developer = DeveloperRiskEngine::assess("Unknown Builder Co", "Phase 3");

        let checklist = LawyerDueDiligenceGenerator::generate(
            &OntarioPropertyType::Condominium,
            &DealType::PreConstruction,
            None,
            Some(&developer),
        );

        assert!(checklist.priority_items.iter().any(|i| i.contains("Tarion")));
        // High risk developer should trigger extra scrutiny
        if developer.overall_risk_score > 0.5 {
            assert!(checklist.priority_items.iter().any(|i| i.contains("Independent legal opinion")));
        }
    }
}
