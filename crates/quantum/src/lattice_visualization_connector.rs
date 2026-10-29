use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::LatticeGridVisualizerWithCorrection;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct LatticeVisualizationConnector;

impl LatticeVisualizationConnector {
    pub async fn connect_visualization_to_pipeline(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Lattice Visualization Connector".to_string());
        }

        let viz_result = LatticeGridVisualizerWithCorrection::visualize_with_correction_overlay(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Lattice Visualization Connector] Grid with overlay connected in {:?}", duration)).await;

        Ok(format!(
            "Lattice Visualization Connector complete | Grid with correction overlay generated and connected\n\n{}",
            viz_result
        ))
    }
}
