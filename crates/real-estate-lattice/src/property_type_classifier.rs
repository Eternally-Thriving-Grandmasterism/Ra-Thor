//! Property Type Classifier for Ontario Real Estate Lattice (RREL)
//! Production-grade classification with OREA form mapping and Thunder Lattice alignment.

use std::collections::HashMap;

/// Ontario property types aligned with OREA and municipal zoning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OntarioPropertyType {
    Detached,
    SemiDetached,
    Townhouse,
    Condominium,
    MultiResidential,
    Commercial,
    Industrial,
    VacantLand,
    Farm,
}

/// Maps property type to recommended OREA forms and key considerations.
pub struct PropertyTypeClassifier;

impl PropertyTypeClassifier {
    /// Returns the primary OREA forms typically used for this property type.
    pub fn recommended_orea_forms(property_type: &OntarioPropertyType) -> Vec<&'static str> {
        match property_type {
            OntarioPropertyType::Detached | OntarioPropertyType::SemiDetached => vec!["Form 100 - Agreement of Purchase and Sale", "Form 101 - Condominium Addendum (if applicable)"],
            OntarioPropertyType::Condominium => vec!["Form 100", "Form 101 - Condominium Addendum", "Status Certificate Review"],
            OntarioPropertyType::Townhouse => vec!["Form 100", "Form 101", "Condo Corp Documents"],
            _ => vec!["Form 100", "Custom Addenda"],
        }
    }

    /// Basic risk flags for the property type.
    pub fn risk_flags(property_type: &OntarioPropertyType) -> Vec<&'static str> {
        match property_type {
            OntarioPropertyType::Condominium => vec!["Status Certificate critical", "Reserve Fund health", "Special Assessments"],
            OntarioPropertyType::Detached => vec!["Inspection priority", "Title search", "Zoning compliance"],
            _ => vec!["Standard due diligence"],
        }
    }
}