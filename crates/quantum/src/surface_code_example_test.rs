use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeMainPipelineRefined;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeExampleTest;

impl SurfaceCodeExampleTest {
    pub async fn run_example_test() -> Result<String, String> {
        let start = Instant::now();

        let request = serde_json::json!({
            "distance": 7,
            "error_rate": 0.01,
            "test_name": "Phase 1 Full Pipeline Example"
        });

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Surface Code Example Test".to_string());
        }

        let pipeline_result = SurfaceCodeMainPipelineRefined::run_complete_refined_pipeline(
            &request,
            CancellationToken::new()
        ).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Example Test] Full example completed in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Example Test complete | Distance: 7 | Error rate: 0.01 | Full pipeline executed successfully\n\n{}",
            pipeline_result
        ))
    }
}
