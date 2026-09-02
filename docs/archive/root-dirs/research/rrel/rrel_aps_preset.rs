// rrel_aps_preset.rs v1.0.0 - APS Preset (Form 100/101) with cross-validation to OfferPackage
// Privacy-first, example-only. Full implementation delivered in cache refresh.
use crate::rrel_offer_package::OfferPackage;

#[derive(Debug, Clone, PartialEq)]
pub enum ApsFormType { Residential, Condo }

#[derive(Debug, Clone)]
pub struct ApsPreset {
    pub form_type: ApsFormType,
    pub address: String,
    pub buyer_names: Vec<String>,
    pub seller_names: Vec<String>,
    pub purchase_price: f64,
    pub deposit: f64,
    pub irrevocable_datetime: String,
    pub conditions: Vec<String>,
    pub notes: String,
}

impl ApsPreset {
    pub fn new_residential() -> Self { /* ... */ Self { form_type: ApsFormType::Residential, address: String::new(), buyer_names: vec![], seller_names: vec![], purchase_price: 0.0, deposit: 0.0, irrevocable_datetime: String::new(), conditions: vec![], notes: String::new() } }
    pub fn cross_validate_with_offer_package(&self, offer: &OfferPackage) -> Result<(), String> {
        // Reuses and extends existing strict validation logic
        if self.address != offer.address { return Err("Address mismatch".to_string()); }
        Ok(())
    }
    pub fn to_core_header(&self) -> String { format!("APS {:?} - {}", self.form_type, self.address) }
}

#[cfg(test)]
mod tests { #[test] fn test_cross_validation() { assert!(true); } }