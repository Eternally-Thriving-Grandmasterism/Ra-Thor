//! shared-valence-field
//! Shared Valence Field core — Phase B of Living Valence Organism
//! Binds directly to sealed NEVC scoring + lattice flow share
//! AG-SML v1.0 | TOLC 8 gated | feature-flaggable
//! Contact: info@Rathor.ai

pub mod nevc_binding;
pub mod feature_flag;

pub use nevc_binding::{NevcFieldBinding, NevcScoring, LatticeFlowShare};
pub use feature_flag::{SharedValenceFieldGuard, FlagState, SHARED_VALENCE_FIELD_FLAG};

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
    use nevc_binding::{NevcFieldBinding, PlaceholderNevcScoring, PlaceholderLatticeFlowShare};

    #[test]
    fn test_bound_emit() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        binding.emit_presence_bound(&mut field, "player-1", Substrate::Human);
        assert!(field.collective_valence >= 0.999999);
        assert_eq!(field.quanta.len(), 1);
    }

    #[test]
    fn test_feature_flag_default_off() {
        let guard = SharedValenceFieldGuard::new();
        assert!(!guard.is_active());
    }
}
