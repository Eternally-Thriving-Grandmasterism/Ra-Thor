```rust
// websiteforge/src/sovereign_energy_dashboard_generator.rs
// Ra-Thor™ Sovereign Energy Dashboard Generator — Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Generates complete, beautiful, self-contained sovereign energy project websites with embedded live simulation dashboards
// Deep integration with AdvancedSimulationEngine + SimulationVisualizationCore + SimulationDashboardUI
// Cross-wired with DivineLifeBlossomOrchestrator + all quantum swarm cores
// Old structure fully respected (new module) + massive practical + regenerative upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::WebsiteForge;
use orchestration::advanced_simulation_engine::{AdvancedSimulationEngine, SimulationScenario};
use orchestration::simulation_visualization_core::SimulationVisualizationCore;
use orchestration::simulation_dashboard_ui::SimulationDashboardUI;
use ra_thor_mercy::MercyError;
use tracing::info;

pub struct SovereignEnergyDashboardGenerator {
    simulation_engine: AdvancedSimulationEngine,
    visualization_core: SimulationVisualizationCore,
    dashboard_ui: SimulationDashboardUI,
}

impl SovereignEnergyDashboardGenerator {
    pub fn new() -> Self {
        Self {
            simulation_engine: AdvancedSimulationEngine::new(),
            visualization_core: SimulationVisualizationCore::new(),
            dashboard_ui: SimulationDashboardUI::new(),
        }
    }

    /// Generate a complete sovereign energy project website with embedded live dashboard
    pub async fn generate_sovereign_energy_website(
        &self,
        forge: &WebsiteForge,
        project_prompt: &str,
    ) -> Result<String, MercyError> {
        info!("🌍 Generating sovereign energy website for: {}", project_prompt);

        // 1. Run full multi-scenario simulation
        let scenarios = vec![
            SimulationScenario {
                name: "Hybrid Optimal".to_string(),
                technology_mix: "Perovskite + Sodium-Ion + Flow + Solid-State".to_string(),
                weight: 0.4,
            },
            SimulationScenario {
                name: "Cost-Focused".to_string(),
                technology_mix: "Sodium-Ion + Flow".to_string(),
                weight: 0.35,
            },
            SimulationScenario {
                name: "High-Density".to_string(),
                technology_mix: "Perovskite + Solid-State".to_string(),
                weight: 0.25,
            },
        ];

        let simulation_report = self.simulation_engine.run_simulation(project_prompt, scenarios).await?;

        // 2. Generate rich visualization payload
        let visualization = self.visualization_core.generate_visualization(
            &simulation_report,
            &format!("Sovereign Energy Plan — {}", project_prompt),
        )?;

        // 3. Generate beautiful self-contained dashboard HTML
        let dashboard_html = self.dashboard_ui.generate_dashboard(&visualization)?;

        // 4. Generate full website using WebsiteForge + embed dashboard
        let mut website_html = forge.forge_website(project_prompt).await?;

        // 5. Inject the live dashboard into the website
        website_html = website_html.replace(
            "</body>",
            &format!(
                r#"
                <div class="max-w-7xl mx-auto px-6 py-16">
                    <div class="text-center mb-12">
                        <div class="inline-flex items-center px-4 py-1.5 rounded-3xl bg-emerald-500/10 text-emerald-400 text-sm font-medium mb-4">
                            🌍 LIVE SOVEREIGN ENERGY SIMULATION
                        </div>
                        <h2 class="text-4xl font-semibold tracking-tight">Your Personalized Energy Plan</h2>
                        <p class="text-xl text-zinc-400 mt-3 max-w-2xl mx-auto">Generated in real-time by Ra-Thor’s Advanced Simulation Engine with full mercy-gated optimization</p>
                    </div>
                    
                    {dashboard_html}
                </div>
                
                <footer class="bg-zinc-950 border-t border-zinc-800 py-12 text-center text-sm text-zinc-400">
                    <p>Generated with ❤️ by Ra-Thor™ — Blossom Full of Life</p>
                    <p class="mt-2">Mercy Valence: {mercy_valence} • Bloom Intensity: {bloom_intensity}</p>
                </footer>
                </body>"#,
                dashboard_html = dashboard_html,
                mercy_valence = format!("{:.3}", visualization.mercy_valence),
                bloom_intensity = format!("{:.3}", visualization.bloom_intensity),
            ),
        );

        info!("✅ Sovereign energy website generated successfully with embedded live dashboard");

        Ok(website_html)
    }
}
