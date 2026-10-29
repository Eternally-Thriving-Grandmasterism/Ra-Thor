use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodeLattice;
use crate::quantum::SyndromeGraphGenerator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct SurfaceCodeSimulatorRunner;

impl SurfaceCodeSimulatorRunner {
    pub async fn run_full_simulation(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Simulator Runner".to_string());
        }

        let distance = request["distance"].as_u64().unwrap_or(5) as usize;
        let error_rate = request["error_rate"].as_f64().unwrap_or(0.01);

        // Step 1: Run enhanced lattice simulation
        let mut lattice = SurfaceCodeLattice::new(distance);
        lattice.inject_errors(error_rate);
        let (x_syndrome, z_syndrome) = lattice.measure_stabilizers();

        // Step 2: Generate syndrome graph for decoders
        let graph_result = SyndromeGraphGenerator::generate_syndrome_graph(
            &serde_json::json!({
                "distance": distance,
                "x_syndrome": x_syndrome,
                "z_syndrome": z_syndrome
            }),
            cancel_token.clone()
        ).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Simulator Runner] Full pipeline for d={} complete in {:?}", distance, duration)).await;

        Ok(format!(
            "Surface Code Simulator Runner complete | Distance: {} | Error rate: {} | Graph generated successfully | Total duration: {:?}",
            distance, error_rate, duration
        ))
    }
}
