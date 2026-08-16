//! resonance_knot.rs
//! Persistent Resonance Knots — NEVC legacy signatures that continue radiating valence
//! Phase E — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

use crate::Epiphany;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A permanent Resonance Knot left in the world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceKnot {
    pub id: String,
    pub origin_epiphany_id: String,
    pub created_at: DateTime<Utc>,
    pub radiating: bool,
    pub cumulative_valence_radiated: f64,
}

impl ResonanceKnot {
    pub fn from_epiphany(epiphany: &Epiphany) -> Self {
        Self {
            id: format!("knot-{}", epiphany.id),
            origin_epiphany_id: epiphany.id.clone(),
            created_at: Utc::now(),
            radiating: true,
            cumulative_valence_radiated: epiphany.valence_gain,
        }
    }

    /// Soft ongoing radiation (called periodically by the lattice)
    pub fn radiate(&mut self, amount: f64) {
        if self.radiating {
            self.cumulative_valence_radiated += amount;
        }
    }
}
