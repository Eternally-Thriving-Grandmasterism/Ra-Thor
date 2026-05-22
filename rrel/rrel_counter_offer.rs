/*!
 * RREL Counter-Offer / Amendment Track (v1.0.0)
 * Professional handling for counter-offers, amendments, and related compliance.
 * Deepened PATSAGi Council integration for reminders and alerts.
 * Aligns with RECO / TRESA disclosure and record-keeping expectations.
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CounterOfferStatus {
    Draft,
    Sent,
    Received,
    Accepted,
    Rejected,
    Withdrawn,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatsagiAlertLevel {
    Info,
    Warning,
    ActionRequired,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterOffer {
    pub id: String,
    pub original_offer_id: String,
    pub property_address: String,
    pub from_party: String,
    pub to_party: String,
    pub changes_summary: String,
    pub new_irrevocable_until: Option<String>,
    pub status: CounterOfferStatus,
    pub created_at: DateTime<Utc>,
    pub notes: Vec<String>,
}

impl CounterOffer {
    pub fn new(
        id: String,
        original_offer_id: String,
        property_address: String,
        from_party: String,
        to_party: String,
        changes_summary: String,
    ) -> Self {
        Self {
            id,
            original_offer_id,
            property_address,
            from_party,
            to_party,
            changes_summary,
            new_irrevocable_until: None,
            status: CounterOfferStatus::Draft,
            created_at: Utc::now(),
            notes: vec![],
        }
    }

    pub fn add_note(&mut self, note: &str) {
        self.notes.push(format!("{}: {}", Utc::now().format("%Y-%m-%d %H:%M"), note));
    }

    pub fn mark_sent(&mut self) {
        self.status = CounterOfferStatus::Sent;
        self.add_note("Marked as sent to other party.");
    }

    pub fn generate_compliance_note(&self) -> String {
        format!(
            "Counter-Offer {} | Status: {:?} | Changes: {} | Retain per O. Reg. 579/05",
            self.id, self.status, self.changes_summary
        )
    }
}

/// Creates a PATSAGi Council reminder with alert level.
pub fn create_patsagi_reminder_for_counter_offer(offer: &CounterOffer) -> String {
    let level = match offer.status {
        CounterOfferStatus::Draft => PatsagiAlertLevel::Info,
        CounterOfferStatus::Sent => PatsagiAlertLevel::Warning,
        CounterOfferStatus::Received => PatsagiAlertLevel::ActionRequired,
        _ => PatsagiAlertLevel::Info,
    };

    format!(
        "[PATSAGi {}] Counter-Offer {} for {} requires follow-up. Current status: {:?}. Mercy-guided action recommended.",
        match level {
            PatsagiAlertLevel::Info => "INFO",
            PatsagiAlertLevel::Warning => "WARNING",
            PatsagiAlertLevel::ActionRequired => "ACTION REQUIRED",
            PatsagiAlertLevel::Critical => "CRITICAL",
        },
        offer.id,
        offer.property_address,
        offer.status
    )
}

/// Generates a structured PATSAGi alert for compliance events.
pub fn generate_patsagi_compliance_alert(
    context: &str,
    level: PatsagiAlertLevel,
    recommended_action: &str,
) -> String {
    format!(
        "[PATSAGi ALERT - {:?}] {} | Recommended: {} | Eternal Mercy Flow active.",
        level, context, recommended_action
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_offer_lifecycle() {
        let mut co = CounterOffer::new(
            "CO-001".to_string(),
            "O-1001".to_string(),
            "456 Example Ave".to_string(),
            "Seller Agent".to_string(),
            "Buyer Agent".to_string(),
            "Price increase + removal of condition".to_string(),
        );
        co.mark_sent();
        assert_eq!(co.status, CounterOfferStatus::Sent);
        assert!(!co.notes.is_empty());
    }

    #[test]
    fn test_patsagi_reminder_generation() {
        let co = CounterOffer::new(
            "CO-042".to_string(),
            "O-2042".to_string(),
            "789 Pine St".to_string(),
            "Listing Agent".to_string(),
            "Buyer Agent".to_string(),
            "Extension of irrevocable period".to_string(),
        );
        let reminder = create_patsagi_reminder_for_counter_offer(&co);
        assert!(reminder.contains("PATSAGi"));
    }
}
