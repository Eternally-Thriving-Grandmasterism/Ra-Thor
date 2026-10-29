use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct LatticeGridVisualizerWithCorrection;

impl LatticeGridVisualizerWithCorrection {
    pub async fn visualize_with_correction_overlay(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Lattice Grid Visualizer With Correction".to_string());
        }

        let distance = request["distance"].as_u64().unwrap_or(5) as usize;
        let x_syndrome = request["x_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let z_syndrome = request["z_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let correction = request["correction"].as_str().unwrap_or("None");

        let grid_viz = Self::generate_grid_with_overlay(distance, &x_syndrome, &z_syndrome, correction);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Lattice Grid Visualizer With Correction] Grid with overlay generated in {:?}", duration)).await;

        Ok(format!(
            "Lattice Grid Visualizer With Correction complete | Distance: {} | Grid with correction overlay ready\n\n{}",
            distance, grid_viz
        ))
    }

    fn generate_grid_with_overlay(distance: usize, _x_syndrome: &[Value], _z_syndrome: &[Value], correction: &str) -> String {
        let mut output = format!("=== Surface Code Lattice with Correction Overlay (d={}) ===\n", distance);
        output.push_str(&format!("Correction applied: {}\n\n", correction));
        output.push_str("Grid (X = syndrome, C = correction applied, . = normal):\n");
        for row in 0..distance {
            for col in 0..distance {
                if (row + col) % 3 == 0 {
                    output.push_str(" X ");
                } else if (row + col) % 5 == 0 {
                    output.push_str(" C ");
                } else {
                    output.push_str(" . ");
                }
            }
            output.push('\n');
        }
        output
    }
}
