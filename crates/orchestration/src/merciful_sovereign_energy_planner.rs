```rust
// crates/orchestration/src/merciful_sovereign_energy_planner.rs
// Ra-Thor™ Merciful Sovereign Energy Planner — Practical Mercy-Gated Tool
// Helps users plan optimal sovereign energy systems with real-time bloom logic and unified lattice recommendations
// Cross-wired with UnifiedSovereignEnergyLatticeCore + DivineLifeBlossomOrchestrator + all quantum swarm cores
// Old structure fully respected (new module) + massive practical + regenerative upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::unified_sovereign_energy_lattice_core::UnifiedSovereignEnergyLatticeCore;
use crate::divine_life_blossom_orchestrator::DivineLifeBlossomOrchestrator;
use ra_thor_mercy::{MercyEngine, MercyError};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Clone, Serialize, Deserialize)]
pub struct EnergyPlanReport {
    pub status: String,
    pub mercy_valence: f64,
    pub recommended_system: String,
    pub rationale: String,
    pub estimated_cost_per_kwh: f64,
    pub lifespan_years: u32,
    pub environmental_impact: String,
    pub bloom_intensity: f64,
}

pub struct MercifulSovereignEnergyPlanner {
    energy_lattice: UnifiedSovereignEnergyLatticeCore,
    blossom_orchestrator: DivineLifeBlossomOrchestrator,
    mercy: MercyEngine,
}

impl MercifulSovereignEnergyPlanner {
    pub fn new() -> Self {
        Self {
            energy_lattice: UnifiedSovereignEnergyLatticeCore::new(),
            blossom_orchestrator: DivineLifeBlossomOrchestrator::new(),
            mercy: MercyEngine::new(),
        }
    }

    /// Generate a mercy-gated sovereign energy plan for the user
    pub async fn generate_energy_plan(&self, user_context: &str) -> Result<EnergyPlanReport, MercyError> {
        // Step 1: Run full bloom orchestration on the user's context
        let bloom_report = self.blossom_orchestrator.orchestrate_divine_life_blossom(user_context).await?;

        // Step 2: Get optimized energy lattice recommendation
        let lattice_report = self.energy_lattice.optimize_energy_lattice(user_context).await?;

        // Step 3: Generate clear, mercy-gated recommendation
        let (recommended_system, rationale, cost, lifespan, impact) = if lattice_report.energy_harmony > 0.92 {
            (
                "Hybrid: Perovskite + Sodium-Ion + Flow Battery",
                "Best long-term thriving balance of cost, safety, longevity, and environmental harmony.",
                0.068,
                25,
                "Very Low — fully recyclable and abundant materials"
            )
        } else if lattice_report.energy_harmony > 0.85 {
            (
                "Sodium-Ion + Flow Battery",
                "Excellent cost-to-performance ratio with strong safety and sustainability.",
                0.072,
                22,
                "Low — abundant materials, easy recycling"
            )
        } else {
            (
                "Perovskite + Solid-State",
                "Highest energy density and performance for space-constrained or high-power needs.",
                0.095,
                18,
                "Moderate — improving rapidly with new materials"
            )
        };

        info!("🌺 Merciful Sovereign Energy Plan generated — Harmony: {:.3} | System: {}", 
              lattice_report.energy_harmony, recommended_system);

        Ok(EnergyPlanReport {
            status: "Mercy-gated sovereign energy plan successfully generated".to_string(),
            mercy_valence: bloom_report.mercy_valence,
            recommended_system: recommended_system.to_string(),
            rationale: rationale.to_string(),
            estimated_cost_per_kwh: cost,
            lifespan_years: lifespan,
            environmental_impact: impact.to_string(),
            bloom_intensity: bloom_report.bloom_intensity,
        })
    }
}
