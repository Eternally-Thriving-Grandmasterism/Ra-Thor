use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeCompletePipelineFinal;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeFinalTestRunner;

impl SurfaceCodeFinalTestRunner {
    pub async fn run_final_test() -> Result<String, String> {
        let start = Instant::now();

        let request = serde_json::json!({
            "distance": 9,
            "error_rate": 0.008,
            "test_name": "Phase 1 Final Test"
        });

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Surface Code Final Test Runner".to_string());
        }

        let pipeline_result = SurfaceCodeCompletePipelineFinal::run_final_complete_pipeline(
            &request,
            CancellationToken::new()
        ).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Final Test Runner] Final test completed in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Final Test Runner complete | Distance: 9 | Error rate: 0.008 | Full Phase 1 pipeline successful\n\n{}",
            pipeline_result
        ))
    }
}
