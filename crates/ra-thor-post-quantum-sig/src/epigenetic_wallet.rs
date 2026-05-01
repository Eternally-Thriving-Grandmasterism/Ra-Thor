//! Epigenetic Wallet State Module — RHPQS v0.1.0 (Enhanced)
//! Ra-Thor Native Feature: Wallets that evolve with CEHI
//! Multi-Generational Signature Inheritance • Advanced Quantum Drift Healing • Mercy-Gated Evolution

use crate::RHPQSKey;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticWalletState {
    pub current_cehi: f64,
    pub generation: u32,
    pub quantum_drift_level: f64,
    pub last_healing: Option<DateTime<Utc>>,
    pub inherited_patterns: Vec<String>,
    pub mercy_influence_history: Vec<f64>,
    pub inherited_signatures: Vec<Vec<u8>>,   // ← NEW: Inherited signatures from previous generations
}

impl EpigeneticWalletState {
    pub fn new(initial_cehi: f64) -> Self {
        Self {
            current_cehi: initial_cehi.max(0.0).min(10.0),
            generation: 1,
            quantum_drift_level: 0.0,
            last_healing: None,
            inherited_patterns: Vec::new(),
            mercy_influence_history: vec![1.0],
            inherited_signatures: Vec::new(),
        }
    }

    /// Inherit from parent generation (now includes signatures)
    pub fn inherit_from_parent(&mut self, parent: &EpigeneticWalletState, parent_signatures: &[Vec<u8>]) {
        self.generation = parent.generation + 1;
        self.current_cehi = (parent.current_cehi * 0.88) + (rand::random::<f64>() * 1.2);
        self.current_cehi = self.current_cehi.max(0.0).min(10.0);

        self.inherited_patterns = parent.inherited_patterns.clone();
        if parent.current_cehi > 7.8 {
            self.inherited_patterns.push("high_mercy_legacy".to_string());
        }

        // Inherit up to 3 signatures from parent
        self.inherited_signatures = parent_signatures.iter().take(3).cloned().collect();

        self.quantum_drift_level = 0.0;
    }

    /// Advanced quantum drift healing (mercy + time based)
    pub fn heal_quantum_drift(&mut self, mercy_valence: f64, time_since_last_healing: f64) {
        let base_healing = mercy_valence * 0.32;
        let time_bonus = (time_since_last_healing * 0.015).min(0.25);
        let total_healing = base_healing + time_bonus;

        self.quantum_drift_level = (self.quantum_drift_level - total_healing).max(0.0);
        self.last_healing = Some(Utc::now());

        if self.quantum_drift_level < 0.08 && mercy_valence > 0.93 {
            self.current_cehi = (self.current_cehi + 0.35).min(10.0);
        }
    }

    /// Evolve wallet state (called periodically)
    pub fn evolve(&mut self, mercy_valence: f64) {
        if mercy_valence > 0.92 {
            self.current_cehi = (self.current_cehi + 0.09).min(10.0);
        }

        self.quantum_drift_level = (self.quantum_drift_level + 0.012).min(1.0);

        if self.quantum_drift_level > 0.55 && mercy_valence > 0.89 {
            self.heal_quantum_drift(mercy_valence, 12.0);
        }

        self.mercy_influence_history.push(mercy_valence);
        if self.mercy_influence_history.len() > 120 {
            self.mercy_influence_history.remove(0);
        }
    }

    /// Check if a signature can be verified using inherited signatures
    pub fn can_verify_with_inherited(&self, signature: &[u8]) -> bool {
        self.inherited_signatures.iter().any(|s| s == signature)
    }

    pub fn get_status_report(&self) -> String {
        format!(
            "Epigenetic Wallet — Gen {} | CEHI: {:.2} | Drift: {:.2} | Inherited Signatures: {} | Last Healing: {:?}",
            self.generation,
            self.current_cehi,
            self.quantum_drift_level,
            self.inherited_signatures.len(),
            self.last_healing.map(|t| t.to_rfc3339())
        )
    }
}
