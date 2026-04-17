use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SurfaceCodeLattice {
    pub distance: usize,
    pub data_qubits: Vec<bool>,      // true = |1⟩, false = |0⟩
    pub measure_qubits_x: Vec<bool>,
    pub measure_qubits_z: Vec<bool>,
}

impl SurfaceCodeLattice {
    pub fn new(distance: usize) -> Self {
        let num_data = distance * distance;
        let num_measure = (distance + 1) * (distance + 1);
        Self {
            distance,
            data_qubits: vec![false; num_data],
            measure_qubits_x: vec![false; num_measure],
            measure_qubits_z: vec![false; num_measure],
        }
    }

    // Measure all X and Z stabilizers
    pub fn measure_stabilizers(&self) -> (Vec<bool>, Vec<bool>) {
        let mut x_syndrome = vec![false; (self.distance + 1) * (self.distance + 1)];
        let mut z_syndrome = vec![false; (self.distance + 1) * (self.distance + 1)];

        // Simplified stabilizer measurement (real version would use actual qubit operators)
        for i in 0..x_syndrome.len() {
            x_syndrome[i] = rand::thread_rng().gen_bool(0.01); // simulated error
            z_syndrome[i] = rand::thread_rng().gen_bool(0.01);
        }
        (x_syndrome, z_syndrome)
    }

    // Inject random Pauli errors for testing
    pub fn inject_errors(&mut self, error_rate: f64) {
        let mut rng = rand::thread_rng();
        for qubit in self.data_qubits.iter_mut() {
            if rng.gen_bool(error_rate) {
                *qubit = !*qubit; // flip bit (simulated Pauli X for simplicity)
            }
        }
    }
}

pub struct SurfaceCodeSimulator;

impl SurfaceCodeSimulator {
    pub async fn run_simulation(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        // Radical Love veto first
        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Surface Code Simulator".to_string());
        }

        let distance = request["distance"].as_u64().unwrap_or(5) as usize;
        let error_rate = request["error_rate"].as_f64().unwrap_or(0.01);

        let mut lattice = SurfaceCodeLattice::new(distance);
        lattice.inject_errors(error_rate);

        let (x_syndrome, z_syndrome) = lattice.measure_stabilizers();

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Surface Code Simulator] d={} simulation complete in {:?}", distance, duration)).await;

        Ok(format!(
            "Surface Code Simulator complete | Distance: {} | Error rate: {} | X syndrome length: {} | Z syndrome length: {} | Duration: {:?}",
            distance, error_rate, x_syndrome.len(), z_syndrome.len(), duration
        ))
    }
}
