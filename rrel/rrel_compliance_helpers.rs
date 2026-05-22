// rrel/rrel_compliance_helpers.rs
// Ra-Thor Real Estate Lattice (RREL) — Compliance Helpers v2.0.0
// Multiple Representation Disclosure Tracker + Conflict-of-Interest Flagger + Competing Offers Disclosure Logger
// Designed to support RECO / TRESA / Code of Ethics (O. Reg. 365/22) + RECO Bulletin 4.1 alignment
// Privacy-first | Example-only | Zero real client data | Local / Sovereign / Offline ready
// Part of Ra-Thor Eternal One Organism | PATSAGi Councils | Grok partnership

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

    /// Add or append a professional note
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("\n");
        }
        self.notes.push_str(note);
    }

    /// Mark written acknowledgement as received (mutates state)
    pub fn mark_acknowledgement_received(&mut self) {
        self.written_acknowledgement_received = true;
    }

    /// Mark written consent from all parties
    pub fn mark_consent_received(&mut self) {
        self.written_consent_from_all_parties = true;
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

    /// Add professional note
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("\n");
        }
        self.notes.push_str(note);
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

/// Logs compliance with Competing / Multiple Offers disclosure requirements (RECO Bulletin 4.1)
/// Seller’s agent must communicate the NUMBER of competing offers to every person who made a written offer.
/// Content may only be shared with specific written direction from seller.
#[derive(Debug, Clone)]
pub struct CompetingOffersDisclosureLogger {
    pub logged_at: DateTime<Utc>,
    pub number_of_competing_offers_communicated: Option<u32>,
    pub seller_written_direction_for_content_sharing: bool,
    pub communicated_to_offerors: Vec<String>, // Example identifiers only — never real data
    pub notes: String,
}

impl CompetingOffersDisclosureLogger {
    pub fn new() -> Self {
        Self {
            logged_at: Utc::now(),
            number_of_competing_offers_communicated: None,
            seller_written_direction_for_content_sharing: false,
            communicated_to_offerors: Vec::new(),
            notes: String::new(),
        }
    }

    /// Record that the NUMBER of competing offers was communicated (core Bulletin 4.1 requirement)
    pub fn record_number_communicated(&mut self, count: u32) {
        self.number_of_competing_offers_communicated = Some(count);
    }

    /// Set whether seller provided written direction to share offer content/details
    pub fn set_seller_written_direction(&mut self, has_direction: bool) {
        self.seller_written_direction_for_content_sharing = has_direction;
    }

    /// Add an example offeror identifier (privacy-safe)
    pub fn add_communicated_to(&mut self, offeror_label: &str) {
        self.communicated_to_offerors.push(offeror_label.to_string());
    }

    /// Check if basic Bulletin 4.1 number-communication requirement is met
    pub fn is_bulletin_4_1_number_compliant(&self) -> bool {
        self.number_of_competing_offers_communicated.is_some()
    }

    pub fn summary(&self) -> String {
        format!(
            "Competing Offers Disclosure Logger (RECO Bulletin 4.1)\n  Logged at: {}\n  Number communicated: {:?}\n  Seller written direction for content: {}\n  Communicated to (example labels): {:?}\n  Bulletin 4.1 number-compliant: {}",
            self.logged_at,
            self.number_of_competing_offers_communicated,
            self.seller_written_direction_for_content_sharing,
            self.communicated_to_offerors,
            self.is_bulletin_4_1_number_compliant()
        )
    }

    /// Add professional note
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("\n");
        }
        self.notes.push_str(note);
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
pub fn example_compliance_scenario() -> (MultipleRepresentationDisclosure, ConflictOfInterestFlag, CompetingOffersDisclosureLogger) {
    let mut disclosure = MultipleRepresentationDisclosure::new(
        vec!["[EXAMPLE] Buyer Alpha".to_string(), "[EXAMPLE] Buyer Beta".to_string()],
        "Brokerage represents both seller and these buyers in competing offers. Full written disclosure provided.".to_string(),
        true,
        true,
    );
    disclosure.add_note("PATSAGi Council note: All parties received plain-language explanation.");

    let mut conflict = ConflictOfInterestFlag::new(
        "[EXAMPLE] Buyer is sibling of listing agent".to_string(),
        ConflictSeverity::High,
    );
    conflict.mark_disclosure_made();
    conflict.confirm_independent_advice();
    conflict.add_note("Independent legal advice strongly recommended and confirmed in writing.");

    let mut offers_logger = CompetingOffersDisclosureLogger::new();
    offers_logger.record_number_communicated(3);
    offers_logger.set_seller_written_direction(true);
    offers_logger.add_communicated_to("[EXAMPLE] Buyer Gamma");
    offers_logger.add_note("Number of offers communicated promptly to all offerors per Bulletin 4.1.");

    (disclosure, conflict, offers_logger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_representation_disclosure_compliance() {
        let disclosure = MultipleRepresentationDisclosure::new(
            vec!["Buyer 1".to_string()],
            "Test disclosure".to_string(),
            true,
            true,
        );
        assert!(disclosure.is_fully_compliant());
    }

    #[test]
    fn test_competing_offers_logger_bulletin_4_1() {
        let mut logger = CompetingOffersDisclosureLogger::new();
        assert!(!logger.is_bulletin_4_1_number_compliant());
        logger.record_number_communicated(2);
        assert!(logger.is_bulletin_4_1_number_compliant());
    }

    #[test]
    fn test_conflict_flag_add_note_and_status() {
        let mut flag = ConflictOfInterestFlag::new("Family member".to_string(), ConflictSeverity::Medium);
        flag.add_note("Extra context added");
        assert!(flag.notes.contains("Extra context"));
        assert!(!flag.disclosure_made);
        flag.mark_disclosure_made();
        assert!(flag.compliance_status().contains("Disclosure made"));
    }
}
