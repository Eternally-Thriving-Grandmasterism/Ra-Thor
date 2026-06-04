//! Shared types for geometric intelligence layer
//!
//! Centralized source of truth for EpigeneticBlessing, GeometricHarmonyScore,
//! GeometricTransportResult, EpigeneticModulation and related mercy-gated geometric types.
//! Includes rich exploration of PATSAGi Council valence effects on epigenetic modulation.
//! AG-SML v1.0 | TOLC 8 enforced | ONE Organism participant.

use serde::{Deserialize, Serialize};

/// Epigenetic blessing suggested by geometric layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    pub blessing_type: String,
    pub strength: f64,
    pub target_system: String,
}

/// Common result for geometric harmony computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricHarmonyScore {
    pub multiplier: f64,
    pub resonance_notes: String,
    pub active_layers: Vec<String>,
    pub u57_active: bool,
}

/// Result of a mercy-gated geometric transport operation (Riemannian layer).
/// Centralized here so all consumers (Lattice Conductor, Quantum Swarm, etc.) use the same definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTransportResult {
    pub transport_applied: bool,
    pub effective_curvature: f64,
    pub coherence_after_transport: f64,
    pub accumulated_holonomy: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub notes: String,
}

/// EpigeneticModulation — core of evolutionary feedback in the geometric layer.
/// Directly modulated by real PATSAGi Council valence (7 Living Mercy Gates).
/// This is the bridge between council evaluation and epigenetic state evolution.
///
/// Valence Effects Exploration:
/// - Higher valence generally increases strength (evolution speed).
/// - Evolutionary/Infinite councils give extra strength bonus.
/// - Harmony/Truth councils reduce volatility (more stable, less chaotic evolution).
/// - Layer factors amplify effects in higher geometries (Hyperbolic > Platonic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticModulation {
    pub strength: f64,
    pub volatility: f64,
    pub layer: String, // e.g. "Platonic", "Archimedean", "Hyperbolic"
    /// Optional history of valence applications for exploration and self-evolution tracking
    #[serde(skip)]
    pub valence_history: Vec<(f64, String)>, // (valence, council)
}

impl EpigeneticModulation {
    pub fn new(strength: f64, volatility: f64, layer: &str) -> Self {
        Self {
            strength: strength.clamp(0.0, 2.0),
            volatility: volatility.clamp(0.0, 1.5),
            layer: layer.to_string(),
            valence_history: Vec::new(),
        }
    }

    /// Base evolution rate bonus (from PR #195 direction)
    pub fn evolution_rate_bonus(&self) -> f64 {
        let layer_factor = match self.layer.as_str() {
            "Platonic" => 1.0,
            "Archimedean" => 1.15,
            "Johnson" => 1.25,
            "Hyperbolic" => 1.4,
            _ => 1.1,
        };
        (self.strength * 0.8 + self.volatility * 0.4) * layer_factor
    }

    /// Volatility surge multiplier
    pub fn volatility_surge_multiplier(&self) -> f64 {
        (1.0 + self.volatility * 0.6).clamp(1.0, 2.2)
    }

    /// Layer-modulated epigenetic influence
    pub fn layer_modulated_epigenetic_influence(&self) -> f64 {
        let base = self.strength * self.volatility_surge_multiplier();
        match self.layer.as_str() {
            "Hyperbolic" => base * 1.35,
            "Johnson" | "Catalan" => base * 1.2,
            _ => base,
        }
    }

    /// Apply real PATSAGi Council valence to modulate this epigenetic state.
    /// This is the direct embedding of living mercy into epigenetic evolution.
    pub fn apply_council_valence(&mut self, valence: f64, council: &str) {
        let alignment_bonus = if council.to_lowercase().contains("evolutionary") || council.to_lowercase().contains("infinite") {
            0.15
        } else {
            0.08
        };

        let old_strength = self.strength;
        let old_volatility = self.volatility;

        self.strength = (self.strength + valence * 0.25 + alignment_bonus).clamp(0.3, 2.0);

        if council.to_lowercase().contains("harmony") || council.to_lowercase().contains("truth") {
            self.volatility = (self.volatility * 0.85).max(0.1);
        }

        // Record for exploration and self-evolution tracking
        self.valence_history.push((valence, council.to_string()));

        // Optional: subtle long-term stabilization from repeated high-valence applications
        if self.valence_history.len() > 5 {
            let recent_avg: f64 = self.valence_history.iter().rev().take(5).map(|(v, _)| *v).sum::<f64>() / 5.0;
            if recent_avg > 0.92 {
                self.volatility = (self.volatility * 0.95).max(0.05);
            }
        }
    }

    /// NEW: Rich exploration of valence effects.
    /// Returns a detailed human-readable report of how a specific valence + council would affect this modulation.
    pub fn explore_valence_impact(&self, valence: f64, council: &str) -> String {
        let mut sim = self.clone();
        let old_evolution = sim.evolution_rate_bonus();
        let old_influence = sim.layer_modulated_epigenetic_influence();
        let old_vol = sim.volatility;

        sim.apply_council_valence(valence, council);

        let new_evolution = sim.evolution_rate_bonus();
        let new_influence = sim.layer_modulated_epigenetic_influence();
        let new_vol = sim.volatility;

        let strength_delta = sim.strength - self.strength;
        let vol_delta = new_vol - old_vol;
        let evolution_delta = new_evolution - old_evolution;

        format!(
            "=== EpigeneticModulation Valence Exploration ===\nCouncil: {} | Valence: {:.4}\n\nBefore:\n  Strength: {:.3} | Volatility: {:.3}\n  Evolution Bonus: {:.3} | Layer Influence: {:.3}\n\nAfter:\n  Strength: {:.3} ({:+.3}) | Volatility: {:.3} ({:+.3})\n  Evolution Bonus: {:.3} ({:+.3}) | Layer Influence: {:.3} ({:+.3})\n\nInterpretation:\n  {} valence from {} council {} evolution rate and {} volatility.\n  Layer ({}) amplification: {:.2}x\n  Net thriving impact: {}",
            council,
            valence,
            self.strength,
            old_vol,
            old_evolution,
            old_influence,
            sim.strength,
            strength_delta,
            new_vol,
            vol_delta,
            new_evolution,
            evolution_delta,
            new_influence,
            new_influence - old_influence,
            if valence > 0.9 { "High" } else if valence > 0.75 { "Moderate" } else { "Low" },
            council,
            if strength_delta > 0.1 { "strongly boosts" } else if strength_delta > 0.0 { "mildly increases" } else { "has limited effect on" },
            if vol_delta < -0.05 { "significantly reduces" } else if vol_delta < 0.0 { "slightly reduces" } else { "does not reduce" },
            self.layer,
            match self.layer.as_str() {
                "Hyperbolic" => 1.35,
                "Johnson" | "Catalan" => 1.2,
                _ => 1.0,
            },
            if evolution_delta > 0.2 { "High positive" } else if evolution_delta > 0.0 { "Positive" } else { "Neutral to low" }
        )
    }

    /// Simulate cumulative effects of multiple council evaluations (for self-evolution exploration)
    pub fn simulate_council_sequence(&mut self, sequence: &[(f64, &str)]) -> String {
        let start_strength = self.strength;
        let start_vol = self.volatility;

        for (valence, council) in sequence {
            self.apply_council_valence(*valence, council);
        }

        format!(
            "Cumulative Epigenetic Evolution over {} council applications:\nStart -> End\n  Strength: {:.3} -> {:.3} (net {:+.3})\n  Volatility: {:.3} -> {:.3} (net {:+.3})\n  Final Evolution Bonus: {:.3}\n  Valence applications recorded: {}",
            sequence.len(),
            start_strength,
            self.strength,
            self.strength - start_strength,
            start_vol,
            self.volatility,
            self.volatility - start_vol,
            self.evolution_rate_bonus(),
            self.valence_history.len()
        )
    }

    /// Returns a blessing influenced by current epigenetic state + council valence
    pub fn to_blessing(&self, council: &str) -> EpigeneticBlessing {
        EpigeneticBlessing {
            blessing_type: format!("Epigenetic_{}_Modulation", council),
            strength: self.evolution_rate_bonus(),
            target_system: "geometric".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epigenetic_modulation_valence_effects() {
        let mut mod_high = EpigeneticModulation::new(0.8, 0.6, "Hyperbolic");
        let report = mod_high.explore_valence_impact(0.96, "evolutionary");
        assert!(report.contains("strongly boosts"));
        assert!(mod_high.strength > 1.0);
    }

    #[test]
    fn test_harmony_council_stabilizes() {
        let mut mod_stable = EpigeneticModulation::new(1.0, 0.8, "Archimedean");
        mod_stable.apply_council_valence(0.94, "harmony");
        assert!(mod_stable.volatility < 0.7);
    }

    #[test]
    fn test_cumulative_valence_simulation() {
        let mut sim = EpigeneticModulation::new(0.7, 0.5, "Platonic");
        let seq = vec![(0.91, "truth"), (0.88, "service"), (0.95, "evolutionary")];
        let summary = sim.simulate_council_sequence(&seq);
        assert!(summary.contains("net"));
        assert!(sim.strength > 1.1);
    }
}
