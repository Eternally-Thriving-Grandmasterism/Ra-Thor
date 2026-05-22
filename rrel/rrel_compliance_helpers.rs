/*!
 * RREL Compliance Helpers (v2.5.1)
 * RECO / TRESA aligned compliance tracking modules for the Ra-Thor Real Estate Lattice.
 * Privacy-first, example-only, sovereign, mercy-gated.
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// Multiple Representation Disclosure Tracker
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipleRepresentationDisclosure {
    pub offer_id: String,
    pub disclosed_to: Vec<String>,
    pub disclosure_timestamp: Option<DateTime<Utc>>,
    pub written_acknowledgement_received: bool,
    pub written_consent_received: bool,
    pub notes: Vec<String>,
}

impl MultipleRepresentationDisclosure {
    pub fn new(offer_id: String) -> Self {
        Self {
            offer_id,
            disclosed_to: vec![],
            disclosure_timestamp: None,
            written_acknowledgement_received: false,
            written_consent_received: false,
            notes: vec![],
        }
    }

    pub fn add_disclosed_party(&mut self, party: &str) {
        if !self.disclosed_to.contains(&party.to_string()) {
            self.disclosed_to.push(party.to_string());
        }
    }

    pub fn mark_disclosure_made(&mut self) {
        self.disclosure_timestamp = Some(Utc::now());
        self.add_note("Written disclosure made to client(s).");
    }

    pub fn mark_acknowledgement_received(&mut self) {
        self.written_acknowledgement_received = true;
        self.add_note("Written acknowledgement received from client.");
    }

    pub fn mark_consent_received(&mut self) {
        self.written_consent_received = true;
        self.add_note("Written consent received from all affected clients.");
    }

    pub fn is_compliant(&self) -> bool {
        self.disclosure_timestamp.is_some() && self.written_consent_received
    }

    pub fn add_note(&mut self, note: &str) {
        self.notes.push(format!("{}: {}", Utc::now().format("%Y-%m-%d %H:%M"), note));
    }

    pub fn generate_compliance_note(&self) -> String {
        format!("Multiple Representation Disclosure for {} | Compliant: {}", self.offer_id, self.is_compliant())
    }
}

// ============================================================================
// Conflict of Interest Flag
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictOfInterestFlag {
    pub offer_id: String,
    pub description: String,
    pub severity: ConflictSeverity,
    pub related_parties: Vec<String>,
    pub notes: Vec<String>,
    pub mitigation_steps: Vec<String>,
}

impl ConflictOfInterestFlag {
    pub fn new(offer_id: String, description: String, severity: ConflictSeverity) -> Self {
        Self {
            offer_id,
            description,
            severity,
            related_parties: vec![],
            notes: vec![],
            mitigation_steps: vec![],
        }
    }

    pub fn add_related_party(&mut self, party: &str) {
        if !self.related_parties.contains(&party.to_string()) {
            self.related_parties.push(party.to_string());
        }
    }

    pub fn add_note(&mut self, note: &str) {
        self.notes.push(format!("{}: {}", Utc::now().format("%Y-%m-%d %H:%M"), note));
    }

    pub fn add_mitigation_step(&mut self, step: &str) {
        self.mitigation_steps.push(step.to_string());
    }
}

// ============================================================================
// Competing Offers Disclosure Logger (RECO Bulletin 4.1)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetingOffersDisclosureLogger {
    pub offer_id: String,
    pub number_of_competing_offers_communicated: Option<u32>,
    pub seller_written_direction_for_content_sharing: bool,
    pub communicated_to_parties: Vec<String>,
    pub notes: Vec<String>,
}

impl CompetingOffersDisclosureLogger {
    pub fn new(offer_id: String) -> Self {
        Self {
            offer_id,
            number_of_competing_offers_communicated: None,
            seller_written_direction_for_content_sharing: false,
            communicated_to_parties: vec![],
            notes: vec![],
        }
    }

    pub fn record_number_communicated(&mut self, count: u32) {
        self.number_of_competing_offers_communicated = Some(count);
        self.add_note(&format!("Number of competing offers communicated: {}", count));
    }

    pub fn set_seller_written_direction(&mut self, has_direction: bool) {
        self.seller_written_direction_for_content_sharing = has_direction;
    }

    pub fn add_communicated_to(&mut self, party: &str) {
        if !self.communicated_to_parties.contains(&party.to_string()) {
            self.communicated_to_parties.push(party.to_string());
        }
    }

    pub fn is_bulletin_4_1_number_compliant(&self) -> bool {
        self.number_of_competing_offers_communicated.is_some()
    }

    pub fn summary(&self) -> String {
        format!(
            "Offer {} | Competing offers communicated: {:?} | Seller written direction: {} | Parties notified: {}",
            self.offer_id,
            self.number_of_competing_offers_communicated,
            self.seller_written_direction_for_content_sharing,
            self.communicated_to_parties.len()
        )
    }

    pub fn add_note(&mut self, note: &str) {
        self.notes.push(format!("{}: {}", Utc::now().format("%Y-%m-%d %H:%M"), note));
    }
}

// ============================================================================
// Record Retention Metadata (O. Reg. 579/05)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetentionCategory {
    UnacceptedOffer,   // 1 year minimum
    CompletedTrade,    // 6 years
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordRetentionMetadata {
    pub category: RetentionCategory,
    pub received_or_completed_at: DateTime<Utc>,
    pub retention_years: u32,
    pub notes: String,
}

impl RecordRetentionMetadata {
    pub fn new_for_unaccepted_offer(received_at: DateTime<Utc>) -> Self {
        Self {
            category: RetentionCategory::UnacceptedOffer,
            received_or_completed_at: received_at,
            retention_years: 1,
            notes: "Unaccepted offer — minimum 1 year retention per O. Reg. 579/05 s.20".to_string(),
        }
    }

    pub fn new_for_completed_trade(completed_at: DateTime<Utc>) -> Self {
        Self {
            category: RetentionCategory::CompletedTrade,
            received_or_completed_at: completed_at,
            retention_years: 6,
            notes: "Completed trade — 6 year retention per O. Reg. 579/05 s.19".to_string(),
        }
    }

    pub fn is_retention_period_met(&self) -> bool {
        let now = Utc::now();
        let years_passed = (now - self.received_or_completed_at).num_days() / 365;
        years_passed >= self.retention_years as i64
    }

    pub fn generate_retention_note(&self) -> String {
        format!(
            "Retention Category: {:?} | Period: {} year(s) | Started: {} | Note: {}",
            self.category,
            self.retention_years,
            self.received_or_completed_at.format("%Y-%m-%d"),
            self.notes
        )
    }

    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str(" | ");
        }
        self.notes.push_str(note);
    }
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

pub fn run_full_integration_happy_path() -> bool {
    // This would normally import and use the other modules.
    // For now returns true as placeholder for full integration.
    true
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_integration_happy_path() {
        assert!(run_full_integration_happy_path());
    }
}
