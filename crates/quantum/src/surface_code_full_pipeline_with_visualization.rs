use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeSimulatorEnhanced;
use crate::quantum::SyndromeGraphGenerator;
use crate::quantum::UnionFindHybridDecoderRefined;
use crate::quantum::MwpmRefinementIntegration;
use crate::quantum::LatticeGridVisualizerWithCorrection;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeFullPipelineWithVisualization;

impl SurfaceCodeFullPipelineWithVisualization {
    pub async fn run_full_visualized_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Full Pipeline With Visualization".to_string());
        }

        // Step 1: Enhanced simulation
        let sim_result = SurfaceCodeSimulatorEnhanced::run_simulation(request, cancel_token.clone()).await?;

        // Step 2: Generate syndrome graph
        let graph_result = SyndromeGraphGenerator::generate_syndrome_graph(request, cancel_token.clone()).await?;

        // Step 3: Run refined hybrid decoder
        let decode_result = UnionFindHybridDecoderRefined::decode_syndrome_refined(request, cancel_token.clone()).await?;

        // Step 4: Apply selective MWPM refinement
        let refinement_result = MwpmRefinementIntegration::apply_mwpm_refinement(request, cancel_token.clone()).await?;

        // Step 5: Generate grid visualization with correction overlay
        let viz_result = LatticeGridVisualizerWithCorrection::visualize_with_correction_overlay(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Full Pipeline With Visualization] Complete visualized pipeline finished in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Full Pipeline With Visualization complete | Simulation: OK | Graph: OK | Hybrid Decode: OK | MWPM Refinement: OK | Visualization: OK | Total duration: {:?}\n\nVisualization:\n{}",
            duration, viz_result
        ))
    }
}
