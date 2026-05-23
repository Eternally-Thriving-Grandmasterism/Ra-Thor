//! GateThresholdMap — Rust prototype mirroring the Lean per-gate decidability model
//! Provides monotonic, local updates to individual gate thresholds (17-24 focus).
//! This is the runtime counterpart to CouncilTuning.lean GateThresholdMap.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GateThresholdMap {
    thresholds: HashMap<String, f64>,
}

impl GateThresholdMap {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // Default thresholds for gates 17-24
        map.insert("ma_at_resonance".to_string(), 0.78);
        map.insert("council_harmony".to_string(), 0.80);
        map.insert("sovereign_legacy".to_string(), 0.80);
        map.insert("infinite_compassion".to_string(), 0.85);
        map.insert("quantum_reverence".to_string(), 0.80);
        map.insert("eternal_recursion".to_string(), 0.82);
        map.insert("cosmic_coherence".to_string(), 0.85);
        map.insert("one_organism_unity".to_string(), 0.90);
        Self { thresholds: map }
    }

    /// Monotonic update: new value must be >= current (enforced)
    pub fn set_threshold(&mut self, gate: &str, new_value: f64) -> Result<f64, String> {
        let current = self.thresholds.get(gate).copied().unwrap_or(0.75);
        if new_value < current {
            return Err(format!("Threshold for '{}' can only increase or stay (current: {:.2})", gate, current));
        }
        self.thresholds.insert(gate.to_string(), new_value);
        Ok(new_value)
    }

    pub fn get_threshold(&self, gate: &str) -> f64 {
        self.thresholds.get(gate).copied().unwrap_or(0.75)
    }

    /// Returns true if the gate would pass after amplification (links to gate_17_24_passes)
    pub fn would_pass(&self, gate: &str, base_score: f64, amplified_score: f64) -> bool {
        amplified_score >= self.get_threshold(gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_increase_only() {
        let mut map = GateThresholdMap::new();
        assert!(map.set_threshold("one_organism_unity", 0.91).is_ok());
        assert!(map.set_threshold("one_organism_unity", 0.89).is_err()); // cannot decrease
    }

    #[test]
    fn test_local_update_does_not_affect_other_gates() {
        let mut map = GateThresholdMap::new();
        let before = map.get_threshold("council_harmony");
        let _ = map.set_threshold("one_organism_unity", 0.93);
        assert_eq!(map.get_threshold("council_harmony"), before);
    }
}