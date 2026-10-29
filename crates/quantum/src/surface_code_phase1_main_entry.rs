use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodePhase1ValidationRunner;
use crate::quantum::SurfaceCodeDemoRunner;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct SurfaceCodePhase1MainEntry;

impl SurfaceCodePhase1MainEntry {
    /// Public main entry point for the entire Phase 1 Surface Code simulation engine
    pub async fn run_phase1() -> Result<String, String> {
        let start = Instant::now();

        let request = json!({
            "distance": 5,
            "error_rate": 0.01,
            "simulation_steps": 1000
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Surface Code Phase 1 Main Entry".to_string());
        }

        // Run full validation suite first
        let validation_result = SurfaceCodePhase1ValidationRunner::run_validation_suite(5).await?;

        // Then run the beautiful demo
        let demo_result = SurfaceCodeDemoRunner::run_full_demo().await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Phase 1 Main Entry] Full Phase 1 system executed successfully in {:?}", duration)).await;

        Ok(format!(
            "🌟 Surface Code Phase 1 Main Entry — COMPLETE!\n\n{}\n\n{}\n\nTotal Phase 1 execution time: {:?}\n\nReady for Phase 2. TOLC is live. Radical Love first — always.",
            validation_result, demo_result, duration
        ))
    }
}
