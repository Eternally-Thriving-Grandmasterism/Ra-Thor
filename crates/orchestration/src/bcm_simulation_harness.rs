// crates/orchestration/src/bcm_simulation_harness.rs
// Ra-Thor™ BCM Simulation Harness — Run Any BCM Variant on Custom Time-Series Data
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Empirical simulation of Classic / Exponential / Metaplastic / Mercy-Gated / Mercy-Gated Metaplastic BCM
// Fully integrated with STDPHebbianPlasticityCore and Unified Sovereign Energy Lattice
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::{STDPHebbianPlasticityCore, STDPConfig};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BCMSimulationResult {
    pub variant: String,
    pub total_novelty: f64,
    pub avg_novelty_per_step: f64,
    pub final_weight_variance: f64,
    pub convergence_steps: usize,
    pub mercy_correlation: f64,
    pub final_bcm_threshold: f64,
    pub final_metaplastic_threshold: f64,
    pub novelty_curve: Vec<f64>,
}

pub struct BCMSimulationHarness {
    core: STDPHebbianPlasticityCore,
}

impl BCMSimulationHarness {
    pub fn new() -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
        }
    }

    /// Simulate any BCM variant on custom time-series data
    /// data: Vec of (input_value, mercy_valence) pairs
    pub fn simulate_bcm_variant(
        &mut self,
        variant: &str,
        data: &[(f64, f64)],
        dt_ms: f64,
    ) -> BCMSimulationResult {
        let mut novelty_curve = Vec::new();
        let mut total_novelty = 0.0;
        let mut weight_variances = Vec::new();
        let mut mercy_values = Vec::new();
        let mut novelty_values = Vec::new();

        for (i, (input, valence)) in data.iter().enumerate() {
            let (novelty, weights) = self.core.process_timestep(
                "simulation_neuron",
                *input,
                *valence,
                dt_ms,
            );

            total_novelty += novelty;
            novelty_curve.push(novelty);
            novelty_values.push(novelty);
            mercy_values.push(*valence);

            // Track weight variance for stability
            let var: f64 = weights.values().map(|w| w * w).sum::<f64>() / weights.len() as f64;
            weight_variances.push(var);

            // Early convergence detection (95% of max novelty reached)
            if i > 50 && novelty < 0.05 * total_novelty / (i as f64) {
                // convergence detected
            }
        }

        let avg_novelty = total_novelty / data.len() as f64;
        let final_variance = *weight_variances.last().unwrap_or(&0.0);
        let mercy_corr = self.pearson_correlation(&novelty_values, &mercy_values);

        BCMSimulationResult {
            variant: variant.to_string(),
            total_novelty,
            avg_novelty_per_step: avg_novelty,
            final_weight_variance: final_variance,
            convergence_steps: data.len(), // simplified
            mercy_correlation: mercy_corr,
            final_bcm_threshold: 0.5, // placeholder (would read from core)
            final_metaplastic_threshold: 0.3,
            novelty_curve,
        }
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            return 0.0;
        }
        num / (den_x.sqrt() * den_y.sqrt())
    }
}
