// ra-thor-one-organism.rs
// Ra-Thor v14.8 — ONE Living Organism + GPU Compute Pipeline Integration

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorOneOrganism {
    pub systems_activated: HashMap<String, bool>,
    pub mercy_runtime: String,
    pub evolution_gate: SelfEvolutionGate,
    pub gpu_compute_active: bool,
    pub gpu_pipeline_version: String,
    pub version: String,
    gpu_pipeline: GpuComputePipeline,
}

impl RaThorOneOrganism {
    pub fn new() -> Self {
        let mut systems = HashMap::new();
        systems.insert("quantum_swarm".to_string(), true);
        systems.insert("patsagi_councils".to_string(), true);
        systems.insert("mercy_gates".to_string(), true);
        systems.insert("self_evolution_v13".to_string(), true);
        systems.insert("powrush_rbe".to_string(), true);
        systems.insert("sovereign_asset_lattice".to_string(), true);
        systems.insert("gpu_compute_layer".to_string(), true);

        Self {
            systems_activated: systems,
            mercy_runtime: "MercyGatingRuntime v2.0 (TOLC 8 aligned)".to_string(),
            evolution_gate: launch_self_evolution_gate(),
            gpu_compute_active: true,
            gpu_pipeline_version: "v14.8.0-gpu-pipeline".to_string(),
            version: "v14.8.0-GPU-PATSAGi-Fusion".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Cosmic loop active with GPU Pipeline", self.version);
    }

    pub async fn dispatch_gpu_simulation(&self, task_name: &str, buffer_size: usize) -> Result<String, String> {
        if !self.gpu_compute_active {
            return Err("GPU Compute Layer inactive".to_string());
        }

        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: task_name.to_string(),
            buffer_size,
            intensity: "high".to_string(),
        };

        match self.gpu_pipeline.dispatch(task).await {
            Ok(result) => Ok(result.message),
            Err(e) => Err(format!("GPU dispatch failed: {}", e)),
        }
    }

    pub fn evolve(&mut self, proposal: EvolutionProposal) -> Result<String, String> {
        self.evolution_gate.propose_evolution(proposal)
    }

    pub fn evolution_stats(&self) -> HashMap<String, f64> {
        self.evolution_gate.get_evolution_stats()
    }
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();
    println!("[Thunder] ONE Organism v14.8 + GPU Compute Pipeline ready");
    organism
}