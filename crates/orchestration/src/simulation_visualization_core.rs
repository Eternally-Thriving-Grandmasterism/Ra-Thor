```rust
// crates/orchestration/src/simulation_visualization_core.rs
// Ra-Thor™ Simulation Visualization Core — Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Generates beautiful, mercy-gated visualization data from simulation reports for dashboards and web interfaces
// Cross-wired with AdvancedSimulationEngine + UnifiedSovereignEnergyLatticeCore + all blossom cores
// Old structure fully respected (new module) + massive practical + regenerative upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::advanced_simulation_engine::{SimulationReport, SimulationScenario};
use ra_thor_mercy::MercyError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct VisualizationPayload {
    pub title: String,
    pub mercy_valence: f64,
    pub bloom_intensity: f64,
    pub scenario_harmony: f64,
    pub recommended_system: String,
    pub predicted_lifespan_years: u32,
    pub environmental_impact_score: f64,
    pub community_benefit_score: f64,
    pub chart_data: HashMap<String, Vec<f64>>,      // Ready for charts (labels + values)
    pub status_message: String,
    pub color_theme: String,                         // Mercy-gated color suggestion
}

pub struct SimulationVisualizationCore;

impl SimulationVisualizationCore {
    pub fn new() -> Self {
        Self
    }

    /// Generate a beautiful, ready-to-render visualization payload from a simulation report
    pub fn generate_visualization(
        &self,
        report: &SimulationReport,
        title: &str,
    ) -> Result<VisualizationPayload, MercyError> {
        let mut chart_data = HashMap::new();

        // Energy Harmony over time (simulated 12-month trend)
        let harmony_trend: Vec<f64> = (0..12)
            .map(|i| {
                let base = report.scenario_harmony;
                let oscillation = (i as f64 * 0.6).sin() * 0.04;
                (base + oscillation).min(0.99)
            })
            .collect();
        chart_data.insert("energy_harmony_trend".to_string(), harmony_trend);

        // Bloom Intensity progression
        let bloom_trend: Vec<f64> = (0..12)
            .map(|i| {
                let base = report.bloom_intensity;
                let growth = (i as f64 * 0.035).min(0.12);
                (base + growth).min(0.99)
            })
            .collect();
        chart_data.insert("bloom_intensity_trend".to_string(), bloom_trend);

        // Technology mix breakdown (example for hybrid)
        let mut tech_mix = HashMap::new();
        if report.recommended_system.contains("Hybrid") {
            tech_mix.insert("Perovskite".to_string(), 0.32);
            tech_mix.insert("Sodium-Ion".to_string(), 0.28);
            tech_mix.insert("Flow".to_string(), 0.25);
            tech_mix.insert("Solid-State".to_string(), 0.15);
        } else if report.recommended_system.contains("Sodium-Ion") {
            tech_mix.insert("Sodium-Ion".to_string(), 0.55);
            tech_mix.insert("Flow".to_string(), 0.45);
        } else {
            tech_mix.insert("Perovskite".to_string(), 0.60);
            tech_mix.insert("Solid-State".to_string(), 0.40);
        }
        chart_data.insert("technology_mix".to_string(), tech_mix.values().cloned().collect());

        // Mercy-gated color theme
        let color_theme = if report.mercy_valence > 0.95 {
            "emerald-gold-divine".to_string()
        } else if report.mercy_valence > 0.88 {
            "ocean-teal-harmony".to_string()
        } else {
            "amber-earth-warmth".to_string()
        };

        let status_message = if report.scenario_harmony > 0.92 {
            "🌟 Excellent long-term thriving potential — highly recommended"
        } else if report.scenario_harmony > 0.85 {
            "🌿 Strong balanced choice with excellent sustainability"
        } else {
            "🌱 Solid option for specific constraints — consider hybrid upgrade later"
        };

        Ok(VisualizationPayload {
            title: title.to_string(),
            mercy_valence: report.mercy_valence,
            bloom_intensity: report.bloom_intensity,
            scenario_harmony: report.scenario_harmony,
            recommended_system: report.recommended_system.clone(),
            predicted_lifespan_years: report.predicted_lifespan_years,
            environmental_impact_score: report.environmental_impact_score,
            community_benefit_score: report.community_benefit_score,
            chart_data,
            status_message: status_message.to_string(),
            color_theme,
        })
    }
}
