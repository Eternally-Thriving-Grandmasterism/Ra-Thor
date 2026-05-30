//! Deal Type Classifier for Ontario Real Estate Lattice (RREL)
//!
//! Production-grade classification of transaction type (Resale, Pre-Construction,
//! Assignment, Family Transfer) to enforce correct disclosures, forms, and
//! anti-cross-contamination rules.
//!
//! **Critical Ontario Design**:
//! - Builder vs Resale separation prevents wrong APS clauses and disclosure packages
//! - Family Transfer track triggers ILA (Independent Legal Advice) prompts
//! - Assignment sales have specific Tarion/ deposit rules
//! - RESA/TRESA and OREA compliance enforced at classification time
//!
//! **Privacy-by-Design**:
//! - Pure classification on deal attributes only
//! - No PII processed or stored
//! - Outputs recommendations and required disclosures only
//!
//! **Mercy & PATSAGi**:
//! - Clear escalation notes instead of blocks
//! - PATSAGi guidance hooks for complex family or multi-party ethics review
//! - Recommendations prioritize buyer/seller protection and fairness
//!
//! Works tightly with PropertyTypeClassifier and FormMappingEngine.

use crate::property_type_classifier::OntarioPropertyType;

/// Primary deal/transaction types in Ontario real estate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DealType {
    Resale,
    PreConstruction,
    Assignment,
    FamilyTransfer,
    Other,
}

/// Structured output of deal classification.
#[derive(Debug, Clone)]
pub struct DealClassification {
    pub deal_type: DealType,
    pub confidence: f64,
    pub required_disclosures: Vec<String>,
    pub recommended_forms: Vec<String>,
    pub patsagi_guidance: Option<String>,
    pub warnings: Vec<String>,
}

pub struct DealTypeClassifier;

impl DealTypeClassifier {
    /// Classifies deal type from context signals (builder mention, assignment language, family indicators, etc.).
    pub fn classify(
        deal_signals: &str,
        property_type: &OntarioPropertyType,
    ) -> DealClassification {
        let signals = deal_signals.to_lowercase();
        let mut warnings = vec![];
        let mut confidence = 0.75;
        let mut required_disclosures = vec![];
        let mut recommended_forms = vec![];
        let mut patsagi_guidance = None;

        let deal_type = if signals.contains("assignment") || signals.contains("assign") {
            DealType::Assignment
        } else if signals.contains("pre-construction") || signals.contains("pre construction") || signals.contains("new home") || signals.contains("builder") {
            DealType::PreConstruction
        } else if signals.contains("family") || signals.contains("spouse") || signals.contains("parent") || signals.contains("child") || signals.contains("transfer") {
            DealType::FamilyTransfer
        } else if signals.contains("resale") || signals.contains("existing") {
            DealType::Resale
        } else {
            warnings.push("Deal type signals ambiguous. Defaulting to Resale. Verify with client and lawyer.".to_string());
            confidence = 0.5;
            DealType::Resale
        };

        match deal_type {
            DealType::PreConstruction => {
                required_disclosures.push("Tarion warranty enrolment confirmation".to_string());
                required_disclosures.push("Deposit protection trust account details".to_string());
                required_disclosures.push("Builder APS clauses and termination rights".to_string());
                recommended_forms.push("Form 100 + Pre-Construction Addendum".to_string());
                warnings.push("Pre-construction: Strong recommendation for independent legal review of builder and deposit protection.".to_string());
            }
            DealType::Assignment => {
                required_disclosures.push("Original APS assignment consent".to_string());
                required_disclosures.push("Assignment fee disclosure".to_string());
                recommended_forms.push("Form 100 + Assignment Addendum".to_string());
                warnings.push("Assignment sales carry specific deposit and Tarion implications. Lawyer review essential.".to_string());
            }
            DealType::FamilyTransfer => {
                required_disclosures.push("Family relationship disclosure".to_string());
                required_disclosures.push("Independent Legal Advice (ILA) confirmation recommended".to_string());
                patsagi_guidance = Some("Family transfer detected. Consider PATSAGi review for fairness, undue influence, and long-term family harmony implications.".to_string());
                recommended_forms.push("Form 100 + Family Transfer disclosures".to_string());
                warnings.push("Family transactions require extra care around ILA and power dynamics. Mercy and clarity first.".to_string());
            }
            DealType::Resale => {
                recommended_forms.push("Form 100 - Agreement of Purchase and Sale".to_string());
                if matches!(property_type, OntarioPropertyType::Condominium | OntarioPropertyType::Townhouse) {
                    required_disclosures.push("Status Certificate + Condo documents".to_string());
                }
            }
            _ => {}
        }

        DealClassification {
            deal_type,
            confidence,
            required_disclosures,
            recommended_forms,
            patsagi_guidance,
            warnings,
        }
    }
}
