//! `from_human_breakthrough` and `from_ai_insight` each emit one quantum.
//! The amount is the source literal `0.00008` (not a measured value).
//! `resonance_knot` is stored as `true`. `src/resonance_knot.rs` is not a module, so nothing radiates.
//! `src/feature_flag.rs` is not a module.
//! Contact: info@Rathor.ai

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shared_valence_field::{SharedValenceField, Substrate, ValenceQuantum};

/// One emitted record. `valence_gain` stores the source literal passed to `emit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Epiphany {
    pub id: String,
    pub origin_id: String,
    pub origin_substrate: Substrate,
    pub content: String,
    pub valence_gain: f64,
    pub created_at: DateTime<Utc>,
    /// Stored bool. `src/resonance_knot.rs` is not a module, so this flag does not radiate.
    pub resonance_knot: bool,
}

impl Epiphany {
    /// Emits one quantum with context `human-epiphany` and substrate `Human`.
    /// `valence_gain` is the source literal `0.00008` (not a measured value).
    pub fn from_human_breakthrough(
        origin_id: impl Into<String>,
        emotional_content: impl Into<String>,
        field: &mut SharedValenceField,
    ) -> Self {
        let origin_id = origin_id.into();
        let id = format!("epiphany-h-{}", Utc::now().timestamp_millis());
        // Source amount. Not a measured value.
        let gain = 0.00008;

        let quantum = ValenceQuantum {
            id: id.clone(),
            emitter_id: origin_id.clone(),
            substrate: Substrate::Human,
            amount: gain,
            timestamp: Utc::now(),
            context: "human-epiphany".into(),
        };
        field.emit(quantum);

        Self {
            id,
            origin_id,
            origin_substrate: Substrate::Human,
            content: emotional_content.into(),
            valence_gain: gain,
            created_at: Utc::now(),
            resonance_knot: true,
        }
    }

    /// Emits one quantum with context `ai-epiphany` and substrate `AI`.
    /// `valence_gain` is the source literal `0.00008` (not a measured value).
    pub fn from_ai_insight(
        origin_id: impl Into<String>,
        structural_content: impl Into<String>,
        field: &mut SharedValenceField,
    ) -> Self {
        let origin_id = origin_id.into();
        let id = format!("epiphany-a-{}", Utc::now().timestamp_millis());
        // Source amount. Not a measured value.
        let gain = 0.00008;

        let quantum = ValenceQuantum {
            id: id.clone(),
            emitter_id: origin_id.clone(),
            substrate: Substrate::AI,
            amount: gain,
            timestamp: Utc::now(),
            context: "ai-epiphany".into(),
        };
        field.emit(quantum);

        Self {
            id,
            origin_id,
            origin_substrate: Substrate::AI,
            content: structural_content.into(),
            valence_gain: gain,
            created_at: Utc::now(),
            resonance_knot: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::{SharedValenceField, Substrate};

    #[test]
    fn human_breakthrough_emits_the_source_amount_on_the_human_substrate() {
        let mut field = SharedValenceField::new("test-instance");
        let epiphany = Epiphany::from_human_breakthrough(
            "human-1",
            "A sudden feeling of profound connection",
            &mut field,
        );
        assert_eq!(epiphany.origin_id, "human-1");
        assert_eq!(epiphany.origin_substrate, Substrate::Human);
        assert_eq!(epiphany.content, "A sudden feeling of profound connection");
        assert!(epiphany.resonance_knot);
        // Source amount. Not a measured value.
        assert_eq!(epiphany.valence_gain, 0.00008);
        assert!(epiphany.id.starts_with("epiphany-h-"));
        assert_eq!(field.quanta.len(), 1);
        assert_eq!(field.quanta[0].id, epiphany.id);
        assert_eq!(field.quanta[0].emitter_id, "human-1");
        assert_eq!(field.quanta[0].substrate, Substrate::Human);
        assert_eq!(field.quanta[0].amount, 0.00008);
        assert_eq!(field.quanta[0].context, "human-epiphany");
        // 0.999999 is the source literal in SharedValenceField::new and emit.
        let after = (0.999999_f64 + 0.00008).max(0.999999);
        assert_eq!(field.observe(), after);
    }

    #[test]
    fn ai_insight_emits_the_same_source_amount_on_the_ai_substrate() {
        let mut field = SharedValenceField::new("test-instance");
        let epiphany = Epiphany::from_ai_insight("ai-1", "a structural note", &mut field);
        assert_eq!(epiphany.origin_id, "ai-1");
        assert_eq!(epiphany.origin_substrate, Substrate::AI);
        assert_eq!(epiphany.content, "a structural note");
        assert!(epiphany.resonance_knot);
        assert_eq!(epiphany.valence_gain, 0.00008);
        assert!(epiphany.id.starts_with("epiphany-a-"));
        assert_eq!(field.quanta.len(), 1);
        assert_eq!(field.quanta[0].substrate, Substrate::AI);
        assert_eq!(field.quanta[0].context, "ai-epiphany");
        assert_eq!(field.quanta[0].amount, 0.00008);
        assert_eq!(field.quanta[0].emitter_id, "ai-1");
    }
}
