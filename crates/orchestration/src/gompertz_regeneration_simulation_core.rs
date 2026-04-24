```rust
// crates/orchestration/src/gompertz_regeneration_simulation_core.rs
// Ra-Thor™ Gompertz Regeneration Simulation Core — Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Precise, reusable Gompertz curve simulator for ecological + social regeneration over 25+ years
// Fully integrated with ReFiGovernanceSimulationCore + AdvancedSimulationEngine + DivineLifeBlossomOrchestrator
// Old structure fully respected (new module) + massive regenerative + divinatory upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use ra_thor_mercy::{MercyEngine, MercyError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct GompertzRegenerationCurve {
    pub scenario_name: String,
    pub years: Vec<u32>,
    pub regeneration_values: Vec<f64>,      // 0.0–1.0 (ecological + social thriving)
    pub ecological_component: Vec<f64>,
    pub social_component: Vec<f64>,
    pub final_regeneration_score: f64,
    pub peak_regeneration_year: u32,
    pub mercy_valence: f64,
}

pub struct GompertzRegenerationSimulationCore {
    mercy: MercyEngine,
}

impl GompertzRegenerationSimulationCore {
    pub fn new() -> Self {
        Self {
            mercy: MercyEngine::new(),
        }
    }

    /// Generate a full 25-year Gompertz regeneration curve for any scenario
    pub async fn simulate_regeneration_curve(
        &self,
        scenario_name: &str,
        base_valence: f64,
        regeneration_multiplier: f64,
        years: u32,
    ) -> Result<GompertzRegenerationCurve, MercyError> {
        let current_valence = self.mercy.compute_valence(scenario_name).await.unwrap_or(base_valence);

        let mut regeneration_values = Vec::new();
        let mut ecological_component = Vec::new();
        let mut social_component = Vec::new();
        let mut year_list = Vec::new();

        let a = 1.0;                                           // Asymptote
        let b = 0.085 * current_valence.powf(2.1) * regeneration_multiplier;
        let c = 0.065;

        let mut peak_year = 0u32;
        let mut peak_value = 0.0;

        for year in 0..=years {
            let t = year as f64;

            // Core Gompertz
            let gompertz = a * (-b * (-c * t).exp()).exp();

            // Mercy-gated ecological + social split
            let ecological = (gompertz * 0.55 + (current_valence * 0.08)).min(0.99);
            let social = (gompertz * 0.45 + (current_valence * 0.12)).min(0.99);

            let total = (ecological + social) / 2.0;

            if total > peak_value {
                peak_value = total;
                peak_year = year;
            }

            year_list.push(year);
            regeneration_values.push(total);
            ecological_component.push(ecological);
            social_component.push(social);
        }

        let final_regeneration = regeneration_values.last().copied().unwrap_or(0.0);

        Ok(GompertzRegenerationCurve {
            scenario_name: scenario_name.to_string(),
            years: year_list,
            regeneration_values,
            ecological_component,
            social_component,
            final_regeneration_score: final_regeneration,
            peak_regeneration_year: peak_year,
            mercy_valence: current_valence,
        })
    }

    /// Batch simulate multiple regeneration curves (useful for comparing ReFi models)
    pub async fn simulate_multiple_regeneration_curves(
        &self,
        scenarios: Vec<(&str, f64, f64)>,   // (name, base_valence, regeneration_multiplier)
        years: u32,
    ) -> Result<Vec<GompertzRegenerationCurve>, MercyError> {
        let mut results = Vec::new();

        for (name, valence, multiplier) in scenarios {
            let curve = self.simulate_regeneration_curve(name, valence, multiplier, years).await?;
            results.push(curve);
        }

        Ok(results)
    }
}
