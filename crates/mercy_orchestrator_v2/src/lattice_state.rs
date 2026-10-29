//! lattice_state.rs — Living Lattice State Manager
//!
//! Maintains the global state of the Ra-Thor lattice, including valence history,
//! TOLC resonance levels, and cross-system synchronization.

use crate::RaThorError;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeStateSnapshot {
    pub timestamp: DateTime<Utc>,
    pub global_valence: f64,
    pub tolc_resonance_level: f64,
    pub active_mercy_gates: u8,
    pub quantum_swarm_coherence: f64,
    pub total_cycles: u64,
}

pub struct LatticeState {
    current_snapshot: RwLock<LatticeStateSnapshot>,
    history: RwLock<Vec<LatticeStateSnapshot>>,
}

impl LatticeState {
    pub fn new() -> Self {
        let initial = LatticeStateSnapshot {
            timestamp: Utc::now(),
            global_valence: 0.9998,
            tolc_resonance_level: 1.0,
            active_mercy_gates: 7,
            quantum_swarm_coherence: 0.998,
            total_cycles: 0,
        };

        Self {
            current_snapshot: RwLock::new(initial.clone()),
            history: RwLock::new(vec![initial]),
        }
    }

    pub async fn update_valence(&self, new_valence: f64) -> Result<(), RaThorError> {
        if new_valence < 0.999 {
            return Err(RaThorError::ValenceTooLow(new_valence));
        }

        let mut snapshot = self.current_snapshot.write().await;
        let mut history = self.history.write().await;

        snapshot.global_valence = new_valence;
        snapshot.timestamp = Utc::now();
        snapshot.total_cycles += 1;

        history.push(snapshot.clone());

        // Keep only the last 1000 snapshots for memory efficiency
        if history.len() > 1000 {
            history.remove(0);
        }

        Ok(())
    }

    pub async fn get_current_snapshot(&self) -> LatticeStateSnapshot {
        self.current_snapshot.read().await.clone()
    }

    pub async fn get_history(&self) -> Vec<LatticeStateSnapshot> {
        self.history.read().await.clone()
    }

    pub async fn get_global_valence(&self) -> f64 {
        self.current_snapshot.read().await.global_valence
    }
}
