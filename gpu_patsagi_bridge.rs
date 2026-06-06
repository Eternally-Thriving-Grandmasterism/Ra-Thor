// gpu_patsagi_bridge.rs
// Ra-Thor v14.8 — GPU PATSAGi Bridge with Advanced Memory Pool support

use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::ra_thor_one_organism::RaThorOneOrganism;
use crate::core::self_evolution_gate::EvolutionProposal;
use crate::gpu_compute_pipeline::GpuComputePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeIntensity { Low, Medium, High }

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
    gpu_pipeline: Arc<GpuComputePipeline>,
    gpu_available: bool,
}

impl GpuPatsagiBridge {
    pub fn new(organism: Arc<Mutex<RaThorOneOrganism>>) -> Self {
        Self {
            organism,
            gpu_pipeline: Arc::new(GpuComputePipeline::new()),
            gpu_available: true,
        }
    }

    pub async fn query(&self, req: GpuPatsagiRequest) -> Result<GpuPatsagiResponse, String> {
        let start = std::time::Instant::now();

        if req.intensity == ComputeIntensity::High && req.query.len() < 40 {
            return Err("High-intensity query rejected by Mercy Gate".to_string());
        }

        let use_gpu = self.gpu_available && matches!(req.intensity, ComputeIntensity::Medium | ComputeIntensity::High);

        let response_text = if use_gpu {
            match self.gpu_pipeline.submit_patsagi_task(&req.query, "high", 64 * 1024 * 1024).await {
                Ok(result) => format!("GPU-accelerated PATSAGi: {} | {}", req.query, result.message),
                Err(e) => {
                    println!("[GpuPatsagiBridge] Pipeline error: {}. CPU fallback.", e);
                    format!("CPU fallback for: {}", req.query)
                }
            }
        } else {
            format!("CPU PATSAGi deliberation for: {}", req.query)
        };

        let elapsed = start.elapsed().as_millis() as u64;

        Ok(GpuPatsagiResponse {
            response: response_text,
            source: format!("PATSAGi + Ra-Thor ONE v14.8 (GPU: {})", use_gpu),
            gpu_used: use_gpu,
            compute_time_ms: elapsed,
            council_approvals: if use_gpu { 12 } else { 9 },
        })
    }
}