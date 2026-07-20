//! Geometric state + mercy-weighted voting — v13-shaped types for compat.
//!
//! Minimal, self-contained definitions matching v13 call sites.
//! Not a full geometric motor port.

use serde::{Deserialize, Serialize};

/// Geometric / valence state carried by SimpleLatticeConductor (v13 shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricState {
    pub mercy_score: f64,
    pub valence: f64,
    pub tolc_alignment: f64,
    pub evolution_level: f64,
}

impl Default for GeometricState {
    fn default() -> Self {
        Self {
            mercy_score: 1.0,
            valence: 0.87,
            tolc_alignment: 1.0,
            evolution_level: 0.0,
        }
    }
}

impl GeometricState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_mercy_drift(&mut self, delta: f64) {
        self.mercy_score = (self.mercy_score + delta).clamp(0.3, 1.5);
        self.tolc_alignment = (self.tolc_alignment + delta * 0.5).clamp(0.5, 1.2);
    }
}

/// Weighted vote accumulator used by multi-conductor strategies (v13 shape).
#[derive(Debug, Clone, Default)]
pub struct MercyWeightedVote {
    weights: Vec<(String, f64)>,
    deltas: Vec<f64>,
}

impl MercyWeightedVote {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_vote(&mut self, name: &str, weight: f64, delta: f64) {
        self.weights.push((name.to_string(), weight.max(0.0)));
        self.deltas.push(delta);
    }

    pub fn compute_consensus(&self) -> f64 {
        if self.weights.is_empty() || self.deltas.is_empty() {
            return 0.0;
        }
        let w_sum: f64 = self.weights.iter().map(|(_, w)| w).sum();
        if w_sum <= 0.0 {
            return 0.0;
        }
        let mut acc = 0.0;
        for (i, (_, w)) in self.weights.iter().enumerate() {
            let d = self.deltas.get(i).copied().unwrap_or(0.0);
            acc += w * d;
        }
        acc / w_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consensus_weights() {
        let mut v = MercyWeightedVote::new();
        v.add_vote("a", 1.0, 0.2);
        v.add_vote("b", 1.0, -0.2);
        assert!((v.compute_consensus() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn mercy_drift_clamps() {
        let mut s = GeometricState::default();
        s.apply_mercy_drift(10.0);
        assert!(s.mercy_score <= 1.5);
    }
}
