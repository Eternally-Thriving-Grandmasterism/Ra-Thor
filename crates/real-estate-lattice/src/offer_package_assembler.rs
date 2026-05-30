//! Offer Package Assembler for Ontario Real Estate Lattice (RREL)
//! Assembles complete, validated offer packages from classified deal + property data.
//! Mercy-gated assembly with clear recommendations.

use crate::deal_type_classifier::DealClassification;
use crate::form_mapping_engine::FormMappingEngine;
use crate::property_type_classifier::OntarioPropertyType;
use crate::offer_package_validator::OfferPackageValidator;

#[derive(Debug, Clone)]
pub struct OfferPackage {
    pub property_type: OntarioPropertyType,
    pub deal_classification: DealClassification,
    pub form_mapping: crate::form_mapping_engine::FormMapping,
    pub validation: crate::offer_package_validator::ValidationResult,
    pub assembled_documents: Vec<String>,
    pub notes: Vec<String>,
}

pub struct OfferPackageAssembler;

impl OfferPackageAssembler {
    pub fn assemble(
        property_type: OntarioPropertyType,
        deal_classification: DealClassification,
        has_signed_aps: bool,
        has_deposit: bool,
        has_status_cert: bool,
    ) -> OfferPackage {
        let form_mapping = FormMappingEngine::map_forms(&property_type, &deal_classification.deal_type);
        let validation = OfferPackageValidator::validate(
            has_signed_aps,
            has_deposit,
            has_status_cert,
            &property_type,
            &deal_classification,
        );

        let mut assembled = vec!["Signed APS (Form 100 base)".to_string()];
        assembled.extend(form_mapping.primary_forms.clone());
        assembled.extend(form_mapping.addenda.clone());
        assembled.extend(form_mapping.supporting_documents.clone());

        let mut notes = vec![];
        if !validation.is_valid {
            notes.push("Validation issues present — review before submission".to_string());
        }
        notes.extend(form_mapping.warnings.clone());

        OfferPackage {
            property_type,
            deal_classification,
            form_mapping,
            validation,
            assembled_documents: assembled,
            notes,
        }
    }
}