use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SyndromeVisualizer;

impl SyndromeVisualizer {
    pub async fn visualize_syndrome_and_correction(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Syndrome Visualizer".to_string());
        }

        let distance = request["distance"].as_u64().unwrap_or(5) as usize;
        let x_syndrome = request["x_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let z_syndrome = request["z_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let correction = request["correction"].as_str().unwrap_or("No correction yet");

        let viz = Self::generate_visualization(distance, &x_syndrome, &z_syndrome, correction);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Syndrome Visualizer] Visualization generated for d={} in {:?}", distance, duration)).await;

        Ok(format!(
            "Syndrome Visualizer complete | Distance: {} | Visualization ready\n\n{}",
            distance, viz
        ))
    }

    fn generate_visualization(distance: usize, x_syndrome: &[Value], z_syndrome: &[Value], correction: &str) -> String {
        format!(
            "=== Surface Code Visualization (d={}) ===\n\
            X-Syndrome length: {}\n\
            Z-Syndrome length: {}\n\
            Correction: {}\n\
            (Full grid visualization will be expanded in next steps)",
            distance, x_syndrome.len(), z_syndrome.len(), correction
        )
    }
}
