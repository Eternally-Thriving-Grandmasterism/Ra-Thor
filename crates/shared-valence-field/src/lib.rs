//! In-memory field.
//! `new` sets `collective_valence` to the source literal 0.999999.
//! `emit` floors with `.max(0.999999)`.
//! That literal is not a measured value.
//! Contact: info@Rathor.ai

pub mod feature_flag;
pub mod nevc_binding;

pub use feature_flag::{FlagState, SharedValenceFieldGuard, SHARED_VALENCE_FIELD_FLAG};
pub use nevc_binding::{LatticeFlowShare, NevcFieldBinding, NevcScoring};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Soft valence quantum — fine-grained NEVC contribution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceQuantum {
    pub id: String,
    pub emitter_id: String,
    pub substrate: Substrate,
    pub amount: f64,
    pub timestamp: DateTime<Utc>,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Substrate {
    Human,
    AI,
}

/// Shared Valence Field state (instance-level)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedValenceField {
    pub instance_id: String,
    pub collective_valence: f64,
    pub quanta: Vec<ValenceQuantum>,
    pub last_updated: DateTime<Utc>,
}

impl SharedValenceField {
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            collective_valence: 0.999999,
            quanta: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    pub fn emit(&mut self, quantum: ValenceQuantum) {
        let new_collective = (self.collective_valence + quantum.amount).max(0.999999);
        self.collective_valence = new_collective;
        self.quanta.push(quantum);
        self.last_updated = Utc::now();
    }

    pub fn observe(&self) -> f64 {
        self.collective_valence
    }

    pub fn emit_presence(emitter_id: impl Into<String>, substrate: Substrate) -> ValenceQuantum {
        ValenceQuantum {
            id: format!("presence-{}", Utc::now().timestamp_millis()),
            emitter_id: emitter_id.into(),
            substrate,
            amount: 0.00001,
            timestamp: Utc::now(),
            context: "presence".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevc_binding::{NevcFieldBinding, PlaceholderLatticeFlowShare, PlaceholderNevcScoring};

    #[test]
    fn test_bound_emit() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding =
            NevcFieldBinding::new(PlaceholderNevcScoring, PlaceholderLatticeFlowShare);

        binding.emit_presence_bound(&mut field, "player-1", Substrate::Human);
        assert_eq!(field.collective_valence, 0.999999 + 0.00001);
        assert_eq!(field.quanta.len(), 1);
    }

    #[test]
    fn emit_floors_at_the_source_literal() {
        let mut field = SharedValenceField::new("floor");
        assert_eq!(field.observe(), 0.999999);
        field.emit(ValenceQuantum {
            id: "q".into(),
            emitter_id: "e".into(),
            substrate: Substrate::AI,
            amount: -10.0,
            timestamp: Utc::now(),
            context: "floor-check".into(),
        });
        assert_eq!(field.observe(), 0.999999);
        assert_eq!(field.quanta.len(), 1);
    }

    #[test]
    fn test_feature_flag_default_off() {
        let guard = SharedValenceFieldGuard::new();
        assert!(!guard.is_active());
    }

    #[test]
    fn guard_stays_off_when_a_criterion_is_false() {
        let mut guard = SharedValenceFieldGuard::new();
        assert!(!guard.try_activate(true, false));
        assert!(!guard.is_active());
        assert!(!guard.try_activate(false, true));
        assert!(guard.try_activate(true, true));
        assert!(guard.is_active());
    }
}
