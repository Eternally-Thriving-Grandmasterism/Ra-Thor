use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeSimulatorEnhanced;
use crate::quantum::UnionFindHybridDecoder;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeDecoderPipeline;

impl SurfaceCodeDecoderPipeline {
    pub async fn run_full_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Decoder Pipeline".to_string());
        }

        // Step 1: Run enhanced simulation
        let sim_result = SurfaceCodeSimulatorEnhanced::run_simulation(request, cancel_token.clone()).await?;

        // Step 2: Run hybrid decoder on the generated syndromes
        let decode_result = UnionFindHybridDecoder::decode_syndrome(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Decoder Pipeline] Full end-to-end pipeline complete in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Decoder Pipeline complete | Simulation: OK | Decoding: OK | Total duration: {:?}",
            duration
        ))
    }
}
