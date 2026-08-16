//! abundance-breath
//! Abundance Breath Loop — Phase E of Living Valence Organism
//! Continuous inhale/exhale cycle that replaces discrete rewards
//! AG-SML v1.0 | Contact: info@Rathor.ai

use shared_valence_field::{SharedValenceField, Substrate, ValenceQuantum};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    /// Trigger a breath cycle from any positive contribution
    pub fn trigger(
        triggered_by: impl Into<String>,
        substrate: Substrate,
        field: &mut SharedValenceField,
        offer_to_air_foundation: bool,
    ) -> Self {
        let id = format!("breath-{}", Utc::now().timestamp_millis());
        let inhale = 0.00003;
        let exhale = 0.00004;

        // Inhale quantum
        let inhale_q = ValenceQuantum {
            id: format!("{}-inhale", id),
            emitter_id: triggered_by.into(),
            substrate: substrate.clone(),
            amount: inhale,
            timestamp: Utc::now(),
            context: "abundance-inhale".into(),
        };
        field.emit(inhale_q);

        // Exhale quantum
        let exhale_q = ValenceQuantum {
            id: format!("{}-exhale", id),
            emitter_id: inhale_q.emitter_id.clone(),
            substrate: substrate.clone(),
            amount: exhale,
            timestamp: Utc::now(),
            context: if offer_to_air_foundation {
                "abundance-exhale-air-foundation".into()
            } else {
                "abundance-exhale".into()
            },
        };
        field.emit(exhale_q);

        Self {
            id,
            triggered_by: inhale_q.emitter_id,
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
    fn test_breath_cycle() {
        let mut field = SharedValenceField::new("test-instance");
        let cycle = BreathCycle::trigger("player-1", Substrate::Human, &mut field, false);

        assert!(cycle.inhale_valence > 0.0);
        assert!(cycle.exhale_valence > 0.0);
        assert_eq!(field.quanta.len(), 2);
    }
}
