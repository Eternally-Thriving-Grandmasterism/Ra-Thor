//! `BreathCycle::trigger` emits two quanta.
//! Inhale amount is the source literal `0.00003`. Exhale amount is the source literal `0.00004`.
//! Those amounts are not measured values.
//! The field floor `0.999999` is applied by `shared-valence-field` (`emit` uses `.max(0.999999)`).
//! `src/feature_flag.rs` is not a module.
//! Contact: info@Rathor.ai

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shared_valence_field::{SharedValenceField, Substrate, ValenceQuantum};

/// One full breath cycle (inhale + exhale)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathCycle {
    pub id: String,
    pub triggered_by: String,
    pub substrate: Substrate,
    pub inhale_valence: f64,
    pub exhale_valence: f64,
    pub offered_to_air_foundation: bool,
    pub created_at: DateTime<Utc>,
}

impl BreathCycle {
    /// Emits an inhale quantum then an exhale quantum.
    /// Amounts are the source literals `0.00003` and `0.00004` (not measured values).
    /// `offer_to_air_foundation` only selects the exhale context string.
    /// No transfer runs in this crate.
    pub fn trigger(
        triggered_by: impl Into<String>,
        substrate: Substrate,
        field: &mut SharedValenceField,
        offer_to_air_foundation: bool,
    ) -> Self {
        let triggered_by = triggered_by.into();
        let id = format!("breath-{}", Utc::now().timestamp_millis());
        // Source amounts. Not measured values.
        let inhale = 0.00003;
        let exhale = 0.00004;

        let inhale_q = ValenceQuantum {
            id: format!("{id}-inhale"),
            emitter_id: triggered_by.clone(),
            substrate: substrate.clone(),
            amount: inhale,
            timestamp: Utc::now(),
            context: "abundance-inhale".into(),
        };
        field.emit(inhale_q);

        let exhale_context = if offer_to_air_foundation {
            "abundance-exhale-air-foundation"
        } else {
            "abundance-exhale"
        };
        let exhale_q = ValenceQuantum {
            id: format!("{id}-exhale"),
            emitter_id: triggered_by.clone(),
            substrate: substrate.clone(),
            amount: exhale,
            timestamp: Utc::now(),
            context: exhale_context.into(),
        };
        field.emit(exhale_q);

        Self {
            id,
            triggered_by,
            substrate,
            inhale_valence: inhale,
            exhale_valence: exhale,
            offered_to_air_foundation: offer_to_air_foundation,
            created_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::{SharedValenceField, Substrate};

    #[test]
    fn trigger_emits_the_two_source_amounts_and_the_plain_exhale_context() {
        let mut field = SharedValenceField::new("test-instance");
        let cycle = BreathCycle::trigger("player-1", Substrate::Human, &mut field, false);

        assert_eq!(cycle.triggered_by, "player-1");
        assert_eq!(cycle.substrate, Substrate::Human);
        assert!(!cycle.offered_to_air_foundation);
        // Source amounts. Not measured values.
        assert_eq!(cycle.inhale_valence, 0.00003);
        assert_eq!(cycle.exhale_valence, 0.00004);
        assert_eq!(field.quanta.len(), 2);
        assert_eq!(field.quanta[0].id, format!("{}-inhale", cycle.id));
        assert_eq!(field.quanta[0].emitter_id, "player-1");
        assert_eq!(field.quanta[0].substrate, Substrate::Human);
        assert_eq!(field.quanta[0].amount, 0.00003);
        assert_eq!(field.quanta[0].context, "abundance-inhale");
        assert_eq!(field.quanta[1].id, format!("{}-exhale", cycle.id));
        assert_eq!(field.quanta[1].emitter_id, "player-1");
        assert_eq!(field.quanta[1].amount, 0.00004);
        assert_eq!(field.quanta[1].context, "abundance-exhale");
        // 0.999999 is the source literal in SharedValenceField::new and emit.
        let after_inhale = (0.999999_f64 + 0.00003).max(0.999999);
        let after_exhale = (after_inhale + 0.00004).max(0.999999);
        assert_eq!(field.observe(), after_exhale);
    }

    #[test]
    fn offer_flag_selects_the_air_foundation_context_and_adds_the_same_amounts() {
        let mut field = SharedValenceField::new("test-instance");
        let cycle = BreathCycle::trigger("player-2", Substrate::AI, &mut field, true);

        assert_eq!(cycle.triggered_by, "player-2");
        assert_eq!(cycle.substrate, Substrate::AI);
        assert!(cycle.offered_to_air_foundation);
        assert_eq!(cycle.inhale_valence, 0.00003);
        assert_eq!(cycle.exhale_valence, 0.00004);
        assert_eq!(field.quanta.len(), 2);
        assert_eq!(field.quanta[0].context, "abundance-inhale");
        assert_eq!(field.quanta[1].context, "abundance-exhale-air-foundation");
        assert_eq!(field.quanta[1].substrate, Substrate::AI);
        assert_eq!(field.quanta[1].amount, 0.00004);
    }
}
