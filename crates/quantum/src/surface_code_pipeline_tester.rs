use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeMainPipelineFinal;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodePipelineTester;

impl SurfaceCodePipelineTester {
    pub async fn run_pipeline_test() -> Result<String, String> {
        let start = Instant::now();

        let request = serde_json::json!({
            "distance": 9,
            "error_rate": 0.008,
            "test_name": "Phase 1 Full Pipeline Test"
        });

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Surface Code Pipeline Tester".to_string());
        }

        let pipeline_result = SurfaceCodeMainPipelineFinal::run_complete_pipeline(
            &request,
            CancellationToken::new()
        ).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Pipeline Tester] Full pipeline test completed in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Pipeline Tester complete | Distance: 9 | Error rate: 0.008 | Full pipeline test successful\n\n{}",
            pipeline_result
        ))
    }
}
