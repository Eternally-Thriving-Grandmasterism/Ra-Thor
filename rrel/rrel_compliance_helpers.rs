// rrel/rrel_compliance_helpers.rs
// Ra-Thor Real Estate Lattice (RREL) — Compliance Helpers v1.0.0
// Multiple Representation Disclosure Tracker + Conflict-of-Interest Flagger
// Designed to support RECO / TRESA / Code of Ethics (O. Reg. 365/22) alignment
// Privacy-first | Example-only | Zero real client data | Local / Sovereign / Offline ready
// Part of Ra-Thor Eternal One Organism | PATSAGi Councils

use chrono::{DateTime, Utc};
use std::fmt;

/// Tracks key details for Multiple Representation disclosures (RECO Bulletin + TRESA)
#[derive(Debug, Clone)]
pub struct MultipleRepresentationDisclosure {
    pub disclosure_made_at: DateTime<Utc>,
    pub disclosed_to: Vec<String>,           // e.g. ["Buyer A", "Buyer B"]
    pub disclosure_content_summary: String,  // Plain language summary of what was disclosed
    pub written_acknowledgement_received: bool,
    pub written_consent_from_all_parties: bool,
    pub notes: String,
}

impl MultipleRepresentationDisclosure {
    pub fn new(
        disclosed_to: Vec<String>,
        disclosure_content_summary: String,
        written_acknowledgement: bool,
        written_consent: bool,
    ) -> Self {
        Self {
            disclosure_made_at: Utc::now(),
            disclosed_to,
            disclosure_content_summary,
            written_acknowledgement_received: written_acknowledgement,
            written_consent_from_all_parties: written_consent,
            notes: String::new(),
        }
    }

    pub fn is_fully_compliant(&self) -> bool {
        self.written_acknowledgement_received && self.written_consent_from_all_parties
    }

    pub fn summary(&self) -> String {
        format!(
            "Multiple Representation Disclosure\n  Made at: {}\n  Disclosed to: {:?}\n  Acknowledgement received: {}\n  Written consent from all: {}\n  Compliant: {}",
            self.disclosure_made_at,
            self.disclosed_to,
            self.written_acknowledgement_received,
            self.written_consent_from_all_parties,
            self.is_fully_compliant()
        )
    }
}

/// Flags potential conflicts of interest, especially useful for FamilyPurchaseAsRealtor track
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct ConflictOfInterestFlag {
    pub flagged_at: DateTime<Utc>,
    pub related_party_description: String,   // e.g. "Buyer is the realtor's brother"
    pub severity: ConflictSeverity,
    pub disclosure_made: bool,
    pub independent_advice_recommended: bool,
    pub independent_advice_confirmed: bool,
    pub notes: String,
}

impl ConflictOfInterestFlag {
    pub fn new(
        related_party_description: String,
        severity: ConflictSeverity,
    ) -> Self {
        Self {
            flagged_at: Utc::now(),
            related_party_description,
            severity,
            disclosure_made: false,
            independent_advice_recommended: true, // Default strong recommendation
            independent_advice_confirmed: false,
            notes: String::new(),
        }
    }

    pub fn mark_disclosure_made(&mut self) {
        self.disclosure_made = true;
    }

    pub fn confirm_independent_advice(&mut self) {
        self.independent_advice_confirmed = true;
    }

    pub fn compliance_status(&self) -> String {
        let mut status = String::new();
        if self.disclosure_made {
            status.push_str("Disclosure made. ");
        } else {
            status.push_str("DISCLOSURE PENDING. ");
        }
        if self.independent_advice_recommended {
            if self.independent_advice_confirmed {
                status.push_str("Independent advice confirmed.");
            } else {
                status.push_str("Strongly recommend independent advice for related party.");
            }
        }
        status
    }
}

/// Simple helper to generate a pre-submission RECO-aligned compliance note
pub fn generate_compliance_note(track: &str, has_conflict: bool) -> String {
    if has_conflict {
        format!(
            "[RECO COMPLIANCE NOTE - {}]\nThis transaction involves a potential conflict of interest.\nEnsure written disclosure + best efforts for independent advice + documented consents.\nAll actions must align with Code of Ethics (O. Reg. 365/22) and TRESA.",
            track
        )
    } else {
        format!(
            "[RECO COMPLIANCE NOTE - {}]\nStandard track. Maintain proper disclosures, record-keeping, and best interests of client.",
            track
        )
    }
}

// Example usage (runtime only)
pub fn example_compliance_scenario() -> (MultipleRepresentationDisclosure, ConflictOfInterestFlag) {
    let disclosure = MultipleRepresentationDisclosure::new(
        vec!["[EXAMPLE] Buyer Alpha".to_string(), "[EXAMPLE] Buyer Beta".to_string()],
        "Brokerage represents both seller and these buyers in competing offers. Full written disclosure provided.".to_string(),
        true,
        true,
    );

    let mut conflict = ConflictOfInterestFlag::new(
        "[EXAMPLE] Buyer is sibling of listing agent".to_string(),
        ConflictSeverity::High,
    );
    conflict.mark_disclosure_made();
    conflict.confirm_independent_advice();

    (disclosure, conflict)
}
