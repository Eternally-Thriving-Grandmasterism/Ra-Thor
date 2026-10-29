/*!
 * rrel_compliance_helpers.rs — Expanded with full PATSAGi scheduling & batch hooks
 * Version: v3.0.0
 */

use chrono::{DateTime, Utc};
use std::collections::HashMap;

// ... (previous structs: MultipleRepresentationDisclosure, ConflictOfInterestFlag, CompetingOffersDisclosureLogger, RecordRetentionMetadata, RetentionCategory remain) ...

#[derive(Debug, Clone, PartialEq)]
pub enum PatsagiAlertLevel {
    Info,
    Warning,
    ActionRequired,
    Critical,
}

#[derive(Debug, Clone)]
pub struct PatsagiScheduledReminder {
    pub id: String,
    pub due_at: DateTime<Utc>,
    pub message: String,
    pub level: PatsagiAlertLevel,
    pub related_counter_offer_id: Option<String>,
    pub acknowledged: bool,
}

impl PatsagiScheduledReminder {
    pub fn new(id: &str, due: DateTime<Utc>, msg: &str, level: PatsagiAlertLevel, related: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            due_at: due,
            message: msg.to_string(),
            level,
            related_counter_offer_id: related,
            acknowledged: false,
        }
    }

    pub fn is_due(&self) -> bool {
        Utc::now() >= self.due_at && !self.acknowledged
    }

    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }
}

pub fn generate_batch_patsagi_alerts(offers: &[super::rrel_counter_offer::CounterOffer]) -> Vec<String> {  // assumes counter_offer module or stub
    let mut alerts = vec![];
    for offer in offers {
        alerts.push(format!("PATSAGi Alert: Counter-offer {} requires review. Level: ActionRequired", offer.id));
    }
    alerts
}

pub fn create_scheduled_patsagi_reminder(
    id: &str,
    due_hours_from_now: i64,
    message: &str,
    level: PatsagiAlertLevel,
) -> PatsagiScheduledReminder {
    let due = Utc::now() + chrono::Duration::hours(due_hours_from_now);
    PatsagiScheduledReminder::new(id, due, message, level, None)
}

// Additional batch and queue helpers can be expanded here for full council coordination.

#[cfg(test)]
mod tests {
    use super::*;
    // ... previous tests + new PATSAGi tests ...
    #[test]
    fn test_patsagi_scheduled_reminder_is_due() {
        let past = Utc::now() - chrono::Duration::hours(1);
        let rem = PatsagiScheduledReminder::new("r1", past, "Test", PatsagiAlertLevel::ActionRequired, None);
        assert!(rem.is_due());
    }
}
