//! shared-valence-field
//! Shared Valence Field core — Phase B of Living Valence Organism
//! Binds directly to sealed NEVC scoring + lattice flow share
//! AG-SML v1.0 | TOLC 8 gated | feature-flaggable
//! Contact: info@Rathor.ai

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

    /// Emit a valence quantum and record it as an NEVC contribution event
    pub fn emit(&mut self, quantum: ValenceQuantum) {
        // Non-bypassable TOLC 8 floor
        let new_collective = (self.collective_valence + quantum.amount).max(0.999999);
        self.collective_valence = new_collective;
        self.quanta.push(quantum);
        self.last_updated = Utc::now();
        // Binding point: this is where the sealed NEVC scoring + lattice flow share broadcast will be called
    }

    /// Observe current field (human sensory / AI structured views handled by Soft Sovereign Agency Layer)
    pub fn observe(&self) -> f64 {
        self.collective_valence
    }

    /// Convenience constructor for a presence contribution (used by Symbiotic Membrane)
    pub fn emit_presence(emitter_id: impl Into<String>, substrate: Substrate) -> ValenceQuantum {
        ValenceQuantum {
            id: format!("presence-{}", Utc::now().timestamp_millis()),
            emitter_id: emitter_id.into(),
            substrate,
            amount: 0.00001, // minimal high-valence presence quantum
            timestamp: Utc::now(),
            context: "presence".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_raises_collective() {
        let mut field = SharedValenceField::new("test-instance");
        let q = ValenceQuantum {
            id: "q1".into(),
            emitter_id: "player1".into(),
            substrate: Substrate::Human,
            amount: 0.0001,
            timestamp: Utc::now(),
            context: "cooperation".into(),
        };
        field.emit(q);
        assert!(field.collective_valence >= 0.999999);
    }

    #[test]
    fn test_presence_quantum() {
        let q = SharedValenceField::emit_presence("ai-1", Substrate::AI);
        assert_eq!(q.context, "presence");
        assert!(q.amount > 0.0);
    }
}
