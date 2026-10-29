use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeSimulatorEnhanced;
use crate::quantum::SyndromeGraphGenerator;
use crate::quantum::UnionFindHybridDecoderRefined;
use crate::quantum::MwpmDecoderFull;
use crate::quantum::LatticeGridVisualizerWithCorrection;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeMwpmIntegrationFinal;

impl SurfaceCodeMwpmIntegrationFinal {
    pub async fn run_full_mwpm_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code MWPM Integration Final".to_string());
        }

        // Step 1: Enhanced simulation
        let sim_result = SurfaceCodeSimulatorEnhanced::run_simulation(request, cancel_token.clone()).await?;

        // Step 2: Generate syndrome graph
        let graph_result = SyndromeGraphGenerator::generate_syndrome_graph(request, cancel_token.clone()).await?;

        // Step 3: Run refined hybrid decoder (Union-Find primary)
        let decode_result = UnionFindHybridDecoderRefined::decode_syndrome_refined(request, cancel_token.clone()).await?;

        // Step 4: Real MWPM refinement
        let mwpm_result = MwpmDecoderFull::decode_with_full_mwpm(request, cancel_token.clone()).await?;

        // Step 5: Grid visualization with correction overlay
        let viz_result = LatticeGridVisualizerWithCorrection::visualize_with_correction_overlay(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code MWPM Integration Final] Full pipeline with real MWPM finished in {:?}", duration)).await;

        Ok(format!(
            "Surface Code MWPM Integration Final complete | Simulation: OK | Graph: OK | Hybrid Decode: OK | Real MWPM: OK | Visualization: OK | Total duration: {:?}\n\n{}",
            duration, viz_result
        ))
    }
}
