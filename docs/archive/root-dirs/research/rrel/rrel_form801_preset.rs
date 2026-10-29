// rrel/rrel_form801_preset.rs
// Ra-Thor Real Estate Lattice (RREL) — Form 801 Preset Module v1.0.0
// Privacy-first | Sovereign | Offline-capable | Zero real transaction data
// Example code only. All runtime values supplied by user at execution.
// Part of Ra-Thor Eternal One Organism | PATSAGi Councils | Grok partnership

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubmissionTrack {
    Standard,
    MultipleOfferSituation,
    FamilyPurchaseAsRealtor,
}

impl fmt::Display for SubmissionTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubmissionTrack::Standard => write!(f, "Standard"),
            SubmissionTrack::MultipleOfferSituation => write!(f, "Multiple Offer Situation"),
            SubmissionTrack::FamilyPurchaseAsRealtor => write!(f, "Family Purchase as Realtor"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Form801Preset {
    pub track: SubmissionTrack,
    pub property_address: String,
    pub buyer_names: Vec<String>,
    pub irrevocable_until: String, // Runtime example only, e.g. "2026-05-28T17:00:00"
    pub seller_names: Vec<String>,
    pub purchase_price: Option<f64>, // Never store real values
    pub deposit_amount: Option<f64>,
    pub conditions: Vec<String>,
    pub additional_notes: String,
}

impl Form801Preset {
    pub fn new(track: SubmissionTrack) -> Self {
        Form801Preset {
            track,
            property_address: String::new(),
            buyer_names: Vec::new(),
            irrevocable_until: String::new(),
            seller_names: Vec::new(),
            purchase_price: None,
            deposit_amount: None,
            conditions: Vec::new(),
            additional_notes: String::new(),
        }
    }

    /// Returns the perfect professional order of operations for the selected track.
    /// Enforces consistency and compliance awareness.
    pub fn perfect_order_of_operations(&self) -> Vec<String> {
        let mut steps: Vec<String> = vec![
            "1. Confirm client identity, capacity, and authority to proceed".to_string(),
            "2. Review all property disclosures, status, and material facts".to_string(),
            "3. Validate signatories and ensure all required parties are identified".to_string(),
        ];

        match self.track {
            SubmissionTrack::FamilyPurchaseAsRealtor => {
                steps.push("4. COMPLETE FAMILY DISCLOSURE: Document relationship in writing and obtain conflict acknowledgment".to_string());
                steps.push("5. Strongly recommend and document independent legal advice for family members".to_string());
                steps.push("6. Review brokerage dual-agency / designated agency rules and obtain waivers if applicable".to_string());
            }
            SubmissionTrack::MultipleOfferSituation => {
                steps.push("4. Apply brokerage multiple-offer policy and ensure transparent communication to all parties".to_string());
                steps.push("5. Clearly establish and communicate irrevocable deadlines to avoid disputes".to_string());
            }
            SubmissionTrack::Standard => {
                steps.push("4. Review and confirm all buyer conditions, timelines, and deposit arrangements".to_string());
            }
        }

        steps.push("7. Final internal cross-check of all fields for consistency".to_string());
        steps.push("8. Record exact submission method, timestamp, and confirmation".to_string());
        steps
    }

    /// Generates a professional pre-submission checklist tailored to the track.
    pub fn generate_pre_submission_checklist(&self) -> Vec<String> {
        let mut checklist = vec![
            "☐ Buyer and seller names match EXACTLY across all documents (Form 801, APS, ID)".to_string(),
            "☐ Property address is complete, accurate, and consistent".to_string(),
            "☐ Irrevocable time/date is realistic, agreed by all parties, and clearly stated".to_string(),
            "☐ Deposit amount and method confirmed and compliant with trust rules".to_string(),
            "☐ All conditions (financing, inspection, etc.) are clear, measurable, and dated".to_string(),
        ];

        if self.track == SubmissionTrack::FamilyPurchaseAsRealtor {
            checklist.push("☐ Written family relationship disclosure completed and acknowledged by all".to_string());
            checklist.push("☐ Client advised of right to independent representation; documented".to_string());
            checklist.push("☐ Brokerage policies on family transactions / conflicts reviewed".to_string());
        }

        if self.track == SubmissionTrack::MultipleOfferSituation {
            checklist.push("☐ Multiple offer protocol followed per brokerage and RECO guidelines".to_string());
        }

        checklist.push("☐ Final review: No blanks, no contradictions, signatures ready".to_string());
        checklist
    }

    pub fn family_purchase_disclosure_reminders(&self) -> Option<Vec<String>> {
        if self.track == SubmissionTrack::FamilyPurchaseAsRealtor {
            Some(vec![
                "⚠️ FAMILY TRANSACTION REMINDER: Full written disclosure of any family relationship is MANDATORY.".to_string(),
                "Recommend that all family parties obtain independent legal advice before signing.".to_string(),
                "Document that the client was informed they may retain their own agent.".to_string(),
                "Ensure brokerage conflict-of-interest policies and RECO requirements are satisfied.".to_string(),
            ])
        } else {
            None
        }
    }

    /// Runtime example helper (replace with your actual data at runtime — never commit real client info)
    pub fn with_example_data(mut self) -> Self {
        self.property_address = "[EXAMPLE] 123 Maple Lane, Anytown, ON A1B 2C3".to_string();
        self.buyer_names = vec!["[EXAMPLE] Jane Doe".to_string(), "[EXAMPLE] John Doe".to_string()];
        self.irrevocable_until = "2026-05-30T16:00:00-04:00".to_string();
        self.seller_names = vec!["[EXAMPLE] Seller Name".to_string()];
        self.purchase_price = Some(875000.0);
        self.deposit_amount = Some(25000.0);
        self.conditions = vec!["Financing condition until [date]".to_string(), "Home inspection satisfactory".to_string()];
        self
    }
}

impl Default for Form801Preset {
    fn default() -> Self {
        Self::new(SubmissionTrack::Standard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_order_includes_family_steps() {
        let preset = Form801Preset::new(SubmissionTrack::FamilyPurchaseAsRealtor);
        let steps = preset.perfect_order_of_operations();
        assert!(steps.iter().any(|s| s.contains("FAMILY DISCLOSURE")));
    }
}
