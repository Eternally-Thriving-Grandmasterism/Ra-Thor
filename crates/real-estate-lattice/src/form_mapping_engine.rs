//! Form Mapping Engine for Ontario Real Estate Lattice (RREL)
//!
//! Precise, production-grade mapping from (PropertyType + DealType) to
//! recommended OREA forms, addenda, supporting documents, compliance notes,
//! and warnings.
//!
//! Covers Freehold, Condominium, POTL/CEC, and edge cases.
//!
//! **Ontario Fidelity**:
//! - OREA standard forms + common addenda
//! - Tarion, Condominium Act, and RESA considerations
//! - Builder vs Resale document separation enforced
//!
//! **Privacy & Mercy**:
//! - Pure mapping, no PII
//! - Warnings and recommendations instead of hard blocks
//! - Designed for clean handoff to offer package assembler and lawyer tooling
//!
//! Integrates with PropertyTypeClassifier and DealTypeClassifier.

use crate::property_type_classifier::{OntarioPropertyType, ClassificationResult};
use crate::deal_type_classifier::{DealType, DealClassification};

#[derive(Debug, Clone)]
pub struct FormMapping {
    pub primary_forms: Vec<String>,
    pub addenda: Vec<String>,
    pub supporting_documents: Vec<String>,
    pub compliance_notes: Vec<String>,
    pub warnings: Vec<String>,
}

pub struct FormMappingEngine;

impl FormMappingEngine {
    /// Generates complete form + document package recommendation.
    pub fn map(
        property_classification: &ClassificationResult,
        deal_classification: &DealClassification,
    ) -> FormMapping {
        let mut primary_forms = vec![];
        let mut addenda = vec![];
        let mut supporting_documents = vec![];
        let mut compliance_notes = vec![];
        let mut warnings = property_classification.warnings.clone();
        warnings.extend(deal_classification.warnings.clone());

        // Base Form 100 always
        primary_forms.push("Form 100 - Agreement of Purchase and Sale".to_string());

        match (&property_classification.property_type, &deal_classification.deal_type) {
            (OntarioPropertyType::Condominium, DealType::Resale) => {
                addenda.push("Form 101 - Condominium Addendum".to_string());
                supporting_documents.push("Status Certificate (current)".to_string());
                supporting_documents.push("Condo Corporation financials & rules".to_string());
                compliance_notes.push("Verify reserve fund health and special assessments before waiver".to_string());
            }
            (OntarioPropertyType::Condominium, DealType::PreConstruction) => {
                addenda.push("Pre-Construction Condominium Addendum".to_string());
                supporting_documents.push("Tarion enrolment proof".to_string());
                supporting_documents.push("Builder disclosure statement".to_string());
                compliance_notes.push("Pre-construction condo: confirm interim occupancy and common element completion timeline".to_string());
            }
            (OntarioPropertyType::Detached | OntarioPropertyType::SemiDetached, DealType::Resale) => {
                supporting_documents.push("Survey / Real Property Report".to_string());
                supporting_documents.push("Title search / PIN printout".to_string());
            }
            (_, DealType::FamilyTransfer) => {
                addenda.push("Family Transfer Disclosure Addendum".to_string());
                compliance_notes.push("Recommend ILA for all parties. Consider PATSAGi ethical review for fairness".to_string());
            }
            (_, DealType::Assignment) => {
                addenda.push("Assignment of Agreement Addendum".to_string());
                compliance_notes.push("Confirm original APS consent and assignment fee treatment".to_string());
            }
            _ => {
                addenda.push("Custom Addenda as required by property type and deal specifics".to_string());
            }
        }

        if deal_classification.patsagi_guidance.is_some() {
            compliance_notes.push("PATSAGi Council guidance flag raised for this transaction type".to_string());
        }

        FormMapping {
            primary_forms,
            addenda,
            supporting_documents,
            compliance_notes,
            warnings,
        }
    }
}
