/*!
 * rrel_aps_preset.rs
 * Ra-Thor Real Estate Lattice (RREL) — Complete APS (Agreement of Purchase and Sale) Preset
 * Form 100 (Residential) / Form 101 (Condo) and related.
 * Version: v1.0.0 — Complete with cross-validation to OfferPackage
 * Part of Ra-Thor Eternal One Organism | PATSAGi Councils
 * Privacy-first, example-only, zero real transaction data, mercy-gated, sovereign.
 */

use chrono::{DateTime, Utc};
use std::fmt;

use super::rrel_form801_preset::{Form801Preset, SubmissionTrack};
use super::rrel_offer_package::{OfferPackage, APSCoreHeader, ValidationError};

#[derive(Debug, Clone, PartialEq)]
pub enum ApsFormType {
    ResidentialForm100,
    CondominiumForm101,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConditionStatus {
    Waived,
    Pending,
    Satisfied,
    NotSatisfied,
}

#[derive(Debug, Clone)]
pub struct ApsCondition {
    pub description: String,
    pub status: ConditionStatus,
    pub due_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct ApsPreset {
    pub form_type: ApsFormType,
    pub property_address: String,
    pub buyer_names: Vec<String>,
    pub seller_names: Vec<String>,
    pub purchase_price: f64,
    pub deposit_amount: f64,
    pub irrevocable_until: DateTime<Utc>,
    pub conditions: Vec<ApsCondition>,
    pub notes: String,
}

impl ApsPreset {
    pub fn new_residential(
        address: &str,
        buyers: Vec<String>,
        sellers: Vec<String>,
        price: f64,
        deposit: f64,
        irrevocable: DateTime<Utc>,
    ) -> Self {
        Self {
            form_type: ApsFormType::ResidentialForm100,
            property_address: address.to_string(),
            buyer_names: buyers,
            seller_names: sellers,
            purchase_price: price,
            deposit_amount: deposit,
            irrevocable_until: irrevocable,
            conditions: vec![],
            notes: String::new(),
        }
    }

    pub fn to_core_header(&self) -> APSCoreHeader {
        APSCoreHeader {
            address: self.property_address.clone(),
            buyer_names: self.buyer_names.clone(),
            irrevocable_time: self.irrevocable_until.to_rfc3339(),
            seller_names: self.seller_names.clone(),
        }
    }

    pub fn add_condition(&mut self, desc: &str, status: ConditionStatus, due: Option<DateTime<Utc>>) {
        self.conditions.push(ApsCondition {
            description: desc.to_string(),
            status,
            due_date: due,
        });
    }

    pub fn cross_validate_with_offer_package(&self, offer_pkg: &OfferPackage) -> Result<(), ValidationError> {
        let core = self.to_core_header();
        // Reuse and extend the existing validation logic
        if core.address.trim().to_lowercase() != offer_pkg.form801.property_address.trim().to_lowercase() {
            return Err(ValidationError::AddressMismatch {
                form801: offer_pkg.form801.property_address.clone(),
                aps: core.address,
            });
        }
        // Buyer names set match
        let offer_buyers: std::collections::HashSet<_> = offer_pkg.form801.buyer_names.iter().map(|s| s.trim().to_lowercase()).collect();
        let aps_buyers: std::collections::HashSet<_> = core.buyer_names.iter().map(|s| s.trim().to_lowercase()).collect();
        if offer_buyers != aps_buyers {
            return Err(ValidationError::BuyerNamesMismatch);
        }
        if core.irrevocable_time.trim() != offer_pkg.form801.irrevocable_until.trim() {
            return Err(ValidationError::IrrevocableTimeMismatch {
                form801: offer_pkg.form801.irrevocable_until.clone(),
                aps: core.irrevocable_time,
            });
        }
        Ok(())
    }

    pub fn generate_summary(&self) -> String {
        format!(
            "APS Preset ({}): Address: {}, Buyers: {:?}, Price: ${:.2}, Irrevocable: {}",
            match self.form_type { ApsFormType::ResidentialForm100 => "Form 100", ApsFormType::CondominiumForm101 => "Form 101", _ => "Other" },
            self.property_address,
            self.buyer_names,
            self.purchase_price,
            self.irrevocable_until
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_aps_creation_and_core_header() {
        let aps = ApsPreset::new_residential(
            "[EXAMPLE] 789 Pine Ave, Town, ON",
            vec!["[EXAMPLE] Jordan Lee".to_string()],
            vec!["[EXAMPLE] Seller Corp".to_string()],
            875000.0,
            50000.0,
            Utc::now(),
        );
        let header = aps.to_core_header();
        assert_eq!(header.address, "[EXAMPLE] 789 Pine Ave, Town, ON");
    }
}
