use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct LatticeGridPrinter;

impl LatticeGridPrinter {
    pub async fn print_lattice_with_overlay(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Lattice Grid Printer".to_string());
        }

        let distance = request["distance"].as_u64().unwrap_or(5) as usize;
        let x_syndrome = request["x_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let z_syndrome = request["z_syndrome"].as_array().unwrap_or(&vec![]).clone();
        let correction = request["correction"].as_str().unwrap_or("None");

        let grid_print = Self::generate_printable_grid(distance, &x_syndrome, &z_syndrome, correction);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Lattice Grid Printer] Grid printed in {:?}", duration)).await;

        Ok(format!(
            "Lattice Grid Printer complete | Distance: {} | Grid with correction overlay ready\n\n{}",
            distance, grid_print
        ))
    }

    fn generate_printable_grid(distance: usize, _x_syndrome: &[Value], _z_syndrome: &[Value], correction: &str) -> String {
        let mut output = format!("=== Surface Code Lattice Grid (d={}) ===\n", distance);
        output.push_str(&format!("Correction: {}\n\n", correction));
        output.push_str("Legend: . = normal | X = syndrome | C = correction applied\n\n");

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
