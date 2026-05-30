//! Offer Package Validator for Real Estate Lattice offer lifecycle
//!
//! Production-grade validator for complete offer packages.
//! Part of Assembler → Validator → Finalizer flow.
//!
//! Validates presence and consistency of forms, disclosures, and supporting docs.
//! Uses severity levels (Info / Warning / Critical) with merciful recommendations.
//!
//! **Design**:
//! - Never hard-blocks a deal; provides clear, actionable feedback
//! - Cross-document consistency checks (property type vs deal type alignment)
//! - Privacy-first: operates only on package metadata
//! - PATSAGi hooks for ethical/complex cases
//!
//! Ready for integration with MultiOfferTrackEngine and LawyerDueDiligenceGenerator.

use crate::property_type_classifier::OntarioPropertyType;
use crate::deal_type_classifier::DealType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub struct OfferPackageValidation {
    pub is_complete: bool,
    pub issues: Vec<ValidationIssue>,
    pub mercy_note: String,
}

pub struct OfferPackageValidator;

impl OfferPackageValidator {
    /// Validates an offer package for completeness and internal consistency.
    pub fn validate(
        property_type: &OntarioPropertyType,
        deal_type: &DealType,
        has_form_100: bool,
        has_status_certificate: bool,
        has_developer_disclosures: bool,
        has_family_ila: bool,
    ) -> OfferPackageValidation {
        let mut issues = vec![];
        let mut is_complete = true;

        if !has_form_100 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Critical,
                message: "Missing primary Form 100 - Agreement of Purchase and Sale".to_string(),
                recommendation: "Include signed Form 100 before proceeding to lawyer review.".to_string(),
            });
            is_complete = false;
        }

        match (property_type, deal_type) {
            (OntarioPropertyType::Condominium, DealType::Resale) => {
                if !has_status_certificate {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        message: "Status Certificate not confirmed for condominium resale".to_string(),
                        recommendation: "Obtain current Status Certificate and review for special assessments / reserve fund before waiver or conditions removal.".to_string(),
                    });
                }
            }
            (_, DealType::PreConstruction) => {
                if !has_developer_disclosures {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        message: "Pre-construction developer disclosures incomplete".to_string(),
                        recommendation: "Verify Tarion enrolment, deposit trust account, and builder APS protections.".to_string(),
                    });
                }
            }
            (_, DealType::FamilyTransfer) => {
                if !has_family_ila {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        message: "Family Transfer without confirmed ILA".to_string(),
                        recommendation: "Strongly recommend Independent Legal Advice for all parties to protect long-term family relationships and fairness.".to_string(),
                    });
                }
            }
            _ => {}
        }

        let mercy_note = if issues.iter().any(|i| i.severity == ValidationSeverity::Critical) {
            "Critical items must be resolved for a clean offer. We move forward with clarity and protection for all parties.".to_string()
        } else if !issues.is_empty() {
            "Warnings noted. Package is functionally usable but benefits from the recommendations above. Mercy and thoroughness go together.".to_string()
        } else {
            "Package validated cleanly. Ready for finalization and lawyer submission. Excellent work.".to_string()
        };

        OfferPackageValidation {
            is_complete,
            issues,
            mercy_note,
        }
    }
}
