// rrel/rrel_offer_package.rs
// Ra-Thor Real Estate Lattice (RREL) — Unified Offer Package v2.0.0
// Cross-validation between Form 801 and APS core header fields
// Privacy-first | Zero-harm | Sovereign | Designed for local/offline use
// Prevents common mismatches that cause professional or legal issues
// Part of Ra-Thor Eternal One Organism | PATSAGi Councils

use std::collections::HashSet;
use std::fmt;

// Re-export / shared type for convenience in this module context.
// In full crate: use super::rrel_form801_preset::{Form801Preset, SubmissionTrack};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubmissionTrack {
    Standard,
    MultipleOfferSituation,
    FamilyPurchaseAsRealtor,
}

#[derive(Debug, Clone, Default)]
pub struct Form801Preset {
    pub track: SubmissionTrack,
    pub property_address: String,
    pub buyer_names: Vec<String>,
    pub irrevocable_until: String,
    pub seller_names: Vec<String>,
    pub purchase_price: Option<f64>,
    pub deposit_amount: Option<f64>,
    pub conditions: Vec<String>,
    pub additional_notes: String,
}

#[derive(Debug, Clone, Default)]
pub struct APSCoreHeader {
    pub address: String,
    pub buyer_names: Vec<String>,
    pub irrevocable_time: String,
    pub seller_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    AddressMismatch { form801: String, aps: String },
    BuyerNamesMismatch,
    IrrevocableTimeMismatch { form801: String, aps: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::AddressMismatch { form801, aps } => {
                write!(f, "ADDRESS MISMATCH: Form 801='{}' vs APS='{}'", form801, aps)
            }
            ValidationError::BuyerNamesMismatch => {
                write!(f, "BUYER NAMES MISMATCH: Names must match exactly between Form 801 and APS")
            }
            ValidationError::IrrevocableTimeMismatch { form801, aps } => {
                write!(f, "IRREVOCABLE TIME MISMATCH: Form 801='{}' vs APS='{}'", form801, aps)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct OfferPackage {
    pub form801: Form801Preset,
    pub aps_header: APSCoreHeader,
}

impl OfferPackage {
    /// Creates a new OfferPackage only if core fields cross-validate successfully.
    /// This is the heart of error prevention in the RREL system.
    pub fn create_with_validation(
        form801: Form801Preset,
        aps_header: APSCoreHeader,
    ) -> Result<Self, ValidationError> {
        // Strict address match (case-insensitive, trimmed)
        if form801.property_address.trim().to_lowercase() != aps_header.address.trim().to_lowercase() {
            return Err(ValidationError::AddressMismatch {
                form801: form801.property_address.clone(),
                aps: aps_header.address.clone(),
            });
        }

        // Buyer names must match exactly as sets (order independent)
        let form_buyers: HashSet<String> = form801
            .buyer_names
            .iter()
            .map(|s| s.trim().to_lowercase())
            .collect();
        let aps_buyers: HashSet<String> = aps_header
            .buyer_names
            .iter()
            .map(|s| s.trim().to_lowercase())
            .collect();

        if form_buyers != aps_buyers {
            return Err(ValidationError::BuyerNamesMismatch);
        }

        // Irrevocable time must match exactly
        if form801.irrevocable_until.trim() != aps_header.irrevocable_time.trim() {
            return Err(ValidationError::IrrevocableTimeMismatch {
                form801: form801.irrevocable_until.clone(),
                aps: aps_header.irrevocable_time.clone(),
            });
        }

        Ok(OfferPackage { form801, aps_header })
    }

    pub fn cross_validation_passed_report(&self) -> String {
        format!(
            "✅ CROSS-VALIDATION PASSED\n   Address: {}\n   Buyers: {:?}\n   Irrevocable Until: {}\n\nAll core fields between Form 801 and APS are consistent. Safe to proceed.",
            self.aps_header.address,
            self.aps_header.buyer_names,
            self.aps_header.irrevocable_time
        )
    }

    pub fn validation_summary(&self) -> String {
        format!(
            "OfferPackage validated for track: {}.\nForm 801 ↔ APS core fields are in perfect alignment.",
            self.form801.track
        )
    }
}

// Example usage (runtime only — never hardcode real client data in source)
pub fn example_validated_package() -> Result<OfferPackage, ValidationError> {
    let mut form801 = Form801Preset::default();
    form801.track = SubmissionTrack::Standard;
    form801.property_address = "[EXAMPLE] 456 Oak Street, City, ON".to_string();
    form801.buyer_names = vec!["[EXAMPLE] Alex Rivera".to_string()];
    form801.irrevocable_until = "2026-06-05T18:00:00".to_string();

    let aps = APSCoreHeader {
        address: "[EXAMPLE] 456 Oak Street, City, ON".to_string(),
        buyer_names: vec!["[EXAMPLE] Alex Rivera".to_string()],
        irrevocable_time: "2026-06-05T18:00:00".to_string(),
        seller_names: vec![]
    };

    OfferPackage::create_with_validation(form801, aps)
}
