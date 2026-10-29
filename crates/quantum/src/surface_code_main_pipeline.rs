use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeSimulatorEnhanced;
use crate::quantum::SyndromeGraphGenerator;
use crate::quantum::UnionFindHybridDecoderRefined;
use crate::quantum::SyndromeVisualizer;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeMainPipeline;

impl SurfaceCodeMainPipeline {
    pub async fn run_complete_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Main Pipeline".to_string());
        }

        // Step 1: Run enhanced simulation
        let sim_result = SurfaceCodeSimulatorEnhanced::run_simulation(request, cancel_token.clone()).await?;

        // Step 2: Generate syndrome graph
        let graph_result = SyndromeGraphGenerator::generate_syndrome_graph(request, cancel_token.clone()).await?;

        // Step 3: Run refined hybrid decoder
        let decode_result = UnionFindHybridDecoderRefined::decode_syndrome_refined(request, cancel_token.clone()).await?;

        // Step 4: Visualize results
        let viz_result = SyndromeVisualizer::visualize_syndrome_and_correction(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Main Pipeline] Complete end-to-end pipeline finished in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Main Pipeline complete | Simulation: OK | Graph: OK | Decoding: OK | Visualization: OK | Total duration: {:?}\n\n{}",
            duration, viz_result
        ))
    }
}
