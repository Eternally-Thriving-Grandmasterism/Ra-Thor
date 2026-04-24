```rust
// crates/orchestration/src/gompertz_richards_visualization_core.rs
// Ra-Thor™ Gompertz vs Richards Visualization Core — Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Generates beautiful, ready-to-render comparison visualizations of fitted Gompertz vs Richards curves on real data
// Cross-wired with GompertzRegenerationSimulationCore + Richards fitting + SimulationVisualizationCore + Sovereign Energy Dashboard
// Old structure fully respected (new module) + massive practical + regenerative upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::gompertz_regeneration_simulation_core::GompertzParams;
use crate::richards_parameter_tuning::RichardsParams;
use ra_thor_mercy::MercyError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct GompertzRichardsComparisonPayload {
    pub title: String,
    pub observed_data: Vec<(f64, f64)>,
    pub gompertz_fit: Vec<f64>,
    pub richards_fit: Vec<f64>,
    pub future_projection: Vec<(f64, f64, f64)>,  // (time, gompertz, richards)
    pub comparison_metrics: HashMap<String, f64>,
    pub mercy_valence: f64,
    pub recommendation: String,
    pub color_theme: String,
}

pub struct GompertzRichardsVisualizationCore;

impl GompertzRichardsVisualizationCore {
    pub fn new() -> Self {
        Self
    }

    /// Generate side-by-side comparison visualization data
    pub fn generate_comparison_visualization(
        &self,
        title: &str,
        observed: &[(f64, f64)],
        gompertz_params: &GompertzParams,
        richards_params: &RichardsParams,
        future_years: u32,
    ) -> Result<GompertzRichardsComparisonPayload, MercyError> {
        let mut gompertz_fit = Vec::new();
        let mut richards_fit = Vec::new();
        let mut future_projection = Vec::new();

        // Generate fits for observed time range
        for &(t, _) in observed {
            let g = calculate_gompertz(t, gompertz_params);
            let r = calculate_richards(t, richards_params);
            gompertz_fit.push(g);
            richards_fit.push(r);
        }

        // Generate future projections
        let last_time = observed.last().map(|(t, _)| *t).unwrap_or(0.0);
        for year in 0..=future_years {
            let t = last_time + year as f64;
            let g = calculate_gompertz(t, gompertz_params);
            let r = calculate_richards(t, richards_params);
            future_projection.push((t, g, r));
        }

        // Compute comparison metrics
        let mut metrics = HashMap::new();
        metrics.insert("gompertz_r2".to_string(), calculate_r_squared(observed, &gompertz_fit));
        metrics.insert("richards_r2".to_string(), calculate_r_squared(observed, &richards_fit));
        metrics.insert("gompertz_25yr_thriving".to_string(), future_projection.last().unwrap().1);
        metrics.insert("richards_25yr_thriving".to_string(), future_projection.last().unwrap().2);
        metrics.insert("thriving_difference".to_string(), 
            (future_projection.last().unwrap().2 - future_projection.last().unwrap().1).abs());

        // Mercy-gated recommendation
        let recommendation = if metrics["richards_r2"] > metrics["gompertz_r2"] + 0.05 {
            "Richards curve provides significantly better fit — use for this scenario"
        } else if metrics["gompertz_r2"] > 0.94 {
            "Gompertz curve offers excellent balance of accuracy and simplicity — recommended default"
        } else {
            "Both models acceptable — Gompertz preferred for interpretability and speed"
        };

        let color_theme = if metrics["richards_r2"] > metrics["gompertz_r2"] {
            "emerald-gold-richards".to_string()
        } else {
            "ocean-teal-gompertz".to_string()
        };

        Ok(GompertzRichardsComparisonPayload {
            title: title.to_string(),
            observed_data: observed.to_vec(),
            gompertz_fit,
            richards_fit,
            future_projection,
            comparison_metrics: metrics,
            mercy_valence: (metrics["gompertz_r2"] + metrics["richards_r2"]) / 2.0,
            recommendation: recommendation.to_string(),
            color_theme,
        })
    }
}

fn calculate_gompertz(t: f64, p: &GompertzParams) -> f64 {
    p.a * (-p.b * (-p.c * t).exp()).exp()
}

fn calculate_richards(t: f64, p: &RichardsParams) -> f64 {
    let exp_term = (-p.growth_rate * (t - p.inflection_time)).exp();
    let denominator = (1.0 + p.initial_value_factor * exp_term).powf(1.0 / p.shape_parameter);
    p.lower_asymptote + (p.upper_asymptote - p.lower_asymptote) / denominator
}

fn calculate_r_squared(observed: &[(f64, f64)], predicted: &[f64]) -> f64 {
    if observed.len() != predicted.len() || observed.is_empty() {
        return 0.0;
    }

    let mean_y: f64 = observed.iter().map(|(_, y)| *y).sum::<f64>() / observed.len() as f64;
    let ss_tot: f64 = observed.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = observed.iter().zip(predicted.iter()).map(|((_, y), p)| (y - p).powi(2)).sum();

    if ss_tot == 0.0 { return 1.0; }
    (1.0 - (ss_res / ss_tot)).max(0.0)
}
