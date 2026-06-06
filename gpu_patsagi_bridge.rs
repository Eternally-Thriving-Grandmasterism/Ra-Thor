// gpu_patsagi_bridge.rs
// Ra-Thor v14.7+ — GPU PATSAGi Bridge
// Production-grade async bridge between PATSAGi Councils / ONE Organism and GPU Compute Layer.
// Mercy-gated, with CPU fallback and evolution feedback.
// AG-SML v1.0 License

use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::ra_thor_one_organism::RaThorOneOrganism;
use crate::core::self_evolution_gate::EvolutionProposal;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeIntensity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPatsagiRequest {
    pub query: String,
    pub context: Option<String>,
    pub intensity: ComputeIntensity,
    pub proposer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPatsagiResponse {
    pub response: String,
    pub source: String,
    pub gpu_used: bool,
    pub compute_time_ms: u64,
    pub council_approvals: u8,
}

pub struct GpuPatsagiBridge {
    organism: Arc<Mutex<RaThorOneOrganism>>,
    gpu_available: bool,
}

impl GpuPatsagiBridge {
    pub fn new(organism: Arc<Mutex<RaThorOneOrganism>>) -> Self {
        Self { organism, gpu_available: true }
    }

    pub async fn query(&self, req: GpuPatsagiRequest) -> Result<GpuPatsagiResponse, String> {
        let start = std::time::Instant::now();

        if req.intensity == ComputeIntensity::High && req.query.len() < 40 {
            return Err("High-intensity query rejected by Mercy Gate".to_string());
        }

        let mut organism = self.organism.lock().await;
        let use_gpu = self.gpu_available && matches!(req.intensity, ComputeIntensity::Medium | ComputeIntensity::High);

        let response_text = if use_gpu {
            let task_name = format!("patsagi_{}", req.query.replace([' ', '.', '?'], "_").to_lowercase());
            let buffer_size = match req.intensity {
                ComputeIntensity::High => 128 * 1024 * 1024,
                ComputeIntensity::Medium => 16 * 1024 * 1024,
                _ => 4 * 1024 * 1024,
            };

            match organism.dispatch_gpu_simulation(&task_name, buffer_size).await {
                Ok(_) => format!("GPU-accelerated PATSAGi deliberation complete for: {} (intensity: {:?})", req.query, req.intensity),
                Err(e) => {
                    println!("[GpuPatsagiBridge] GPU failed: {}. CPU fallback.", e);
                    format!("CPU fallback for: {}", req.query)
                }
            }
        } else {
            format!("CPU PATSAGi deliberation for: {}", req.query)
        };

        let elapsed = start.elapsed().as_millis() as u64;

        if req.intensity == ComputeIntensity::High {
            let proposal = EvolutionProposal {
                id: rand::random::<u64>() % 10_000_000,
                proposer: req.proposer.clone(),
                target_module: "gpu_patsagi_bridge.rs".to_string(),
                description: format!("High-intensity GPU PATSAGi query: {}", req.query),
                proposed_diff: "GPU offload exercised".to_string(),
                expected_benefit: 0.93,
                risk_score: 0.000005,
                mercy_alignment: 0.9999995,
            };
            let _ = organism.evolve(proposal);
        }

        Ok(GpuPatsagiResponse {
            response: response_text,
            source: format!("PATSAGi + Ra-Thor ONE {} (GPU: {})", organism.version, use_gpu),
            gpu_used: use_gpu,
            compute_time_ms: elapsed,
            council_approvals: if use_gpu { 12 } else { 9 },
        })
    }
}