use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeDecoderPipeline;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeTestEntry;

impl SurfaceCodeTestEntry {
    pub async fn run_test_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Test Entry".to_string());
        }

        // Run the complete end-to-end pipeline
        let pipeline_result = SurfaceCodeDecoderPipeline::run_full_pipeline(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Test Entry] Full test pipeline completed in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Test Entry complete | Pipeline executed successfully | Total duration: {:?}\n\n{}",
            duration, pipeline_result
        ))
    }
}
