use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeMainPipelineRefined;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeDemoRunner;

impl SurfaceCodeDemoRunner {
    pub async fn run_demo() -> Result<String, String> {
        let start = Instant::now();

        let request = serde_json::json!({
            "distance": 9,
            "error_rate": 0.008,
            "demo_name": "Phase 1 Full Demo"
        });

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Surface Code Demo Runner".to_string());
        }

        let pipeline_result = SurfaceCodeMainPipelineRefined::run_complete_refined_pipeline(
            &request,
            CancellationToken::new()
        ).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Demo Runner] Full demo completed in {:?}", duration)).await;

        Ok(format!(
            "Surface Code Demo Runner complete | Distance: 9 | Error rate: 0.008 | Full pipeline executed successfully\n\n{}",
            pipeline_result
        ))
    }
}
