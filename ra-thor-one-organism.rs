/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.10 — ONE Living Organism with PATSAGi Council Decision Logic

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask};

// === Council Readiness & Decision Types (integrated from PATSAGi governance layer) ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilReadinessMetrics {
    pub council_ready: bool,
    pub mercy_norm: f64,
    pub suggested_confidence_delta: f64,
    pub evolution_level: u32,
    pub last_updated_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CouncilDecision {
    ApproveEvolution { confidence_boost: f64 },
    RejectEvolution { reason: String },
    AdjustRbeParameters { resource_flow_multiplier: f64, council_influence: f64 },
    RequestAdditionalGpuResources { buffer_size_increase: usize },
    EmergencyMercyIntervention { severity: f64 },
    NoAction,
}

/// Lightweight PATSAGi Council decision engine embedded in the ONE Organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatsagiCouncil {
    mercy_norm_threshold: f64,
    council_ready_threshold: f64,
}

impl PatsagiCouncil {
    pub fn new() -> Self {
        Self {
            mercy_norm_threshold: 0.75,
            council_ready_threshold: 0.6,
        }
    }

    pub fn decide(&self, metrics: &CouncilReadinessMetrics) -> CouncilDecision {
        if !metrics.council_ready {
            return CouncilDecision::RejectEvolution {
                reason: "Council not ready (mercy gates not satisfied)".to_string(),
            };
        }

        if metrics.mercy_norm >= self.mercy_norm_threshold {
            return CouncilDecision::ApproveEvolution {
                confidence_boost: metrics.suggested_confidence_delta.max(0.05),
            };
        }

        if metrics.mercy_norm < 0.4 {
            return CouncilDecision::EmergencyMercyIntervention {
                severity: (0.4 - metrics.mercy_norm) * 2.0,
            };
        }

        if metrics.suggested_confidence_delta > 0.15 {
            return CouncilDecision::RequestAdditionalGpuResources {
                buffer_size_increase: 2048,
            };
        }

        CouncilDecision::AdjustRbeParameters {
            resource_flow_multiplier: 1.0 + (metrics.mercy_norm - 0.5) * 0.5,
            council_influence: metrics.mercy_norm,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorOneOrganism {
    pub systems_activated: HashMap<String, bool>,
    pub mercy_runtime: String,
    pub evolution_gate: SelfEvolutionGate,
    pub gpu_compute_active: bool,
    pub gpu_pipeline_version: String,
    pub version: String,
    gpu_pipeline: GpuComputePipeline,

    // === NEW: PATSAGi Council + Metrics State ===
    patsagi_council: PatsagiCouncil,
    last_council_metrics: Option<CouncilReadinessMetrics>,
    council_tick: u64,
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
            gpu_pipeline_version: "v14.10.0-gpu-patsagi-council".to_string(),
            version: "v14.10.0-ONE-Organism-PATSAGi-Council".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),

            // PATSAGi Council integration
            patsagi_council: PatsagiCouncil::new(),
            last_council_metrics: None,
            council_tick: 0,
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Cosmic loop with PATSAGi Council Decision Logic", self.version);
    }

    /// Feed a GPU audit into PATSAGi Council decision logic and return the resulting governance decision
    pub fn feed_gpu_audit_into_council_decision(
        &mut self,
        mercy_norm: f64,
        council_ready: bool,
        suggested_confidence_delta: f64,
    ) -> CouncilDecision {
        self.council_tick += 1;

        let metrics = CouncilReadinessMetrics {
            council_ready,
            mercy_norm,
            suggested_confidence_delta,
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        // Execute side effects based on decision (extend as needed)
        match &decision {
            CouncilDecision::ApproveEvolution { confidence_boost } => {
                println!("[ONE Organism] PATSAGi Council APPROVED evolution (+{:.4} confidence)", confidence_boost);
            }
            CouncilDecision::AdjustRbeParameters { resource_flow_multiplier, council_influence } => {
                println!(
                    "[ONE Organism] PATSAGi Council ADJUST RBE (flow x{:.2}, influence {:.2})",
                    resource_flow_multiplier, council_influence
                );
            }
            CouncilDecision::RequestAdditionalGpuResources { buffer_size_increase } => {
                println!("[ONE Organism] PATSAGi Council REQUESTS more GPU (+{} buffer)", buffer_size_increase);
            }
            CouncilDecision::EmergencyMercyIntervention { severity } => {
                println!("[ONE Organism] PATSAGi Council EMERGENCY MERCY (severity {:.2})", severity);
            }
            CouncilDecision::RejectEvolution { reason } => {
                println!("[ONE Organism] PATSAGi Council REJECTED: {}", reason);
            }
            CouncilDecision::NoAction => {}
        }

        decision
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
            Err(e) => Err(e),
        }
    }

    /// New: Dispatch GPU task and immediately feed result into PATSAGi Council
    pub async fn dispatch_gpu_and_feed_council(
        &mut self,
        task_name: &str,
        buffer_size: usize,
    ) -> Result<(String, CouncilDecision), String> {
        let message = self.dispatch_gpu_simulation(task_name, buffer_size).await?;

        // In real integration the GPU dispatch would return a full MercyGpuAudit.
        // Here we synthesize a representative audit from the task for demonstration.
        let mercy_norm = 0.82; // placeholder — replace with real audit.mercy_norm
        let council_ready = true;
        let suggested_confidence_delta = 0.12;

        let decision = self.feed_gpu_audit_into_council_decision(
            mercy_norm,
            council_ready,
            suggested_confidence_delta,
        );

        Ok((message, decision))
    }

    pub async fn get_gpu_memory_stats(&self) -> crate::gpu_compute_pipeline::GpuMemoryStats {
        self.gpu_pipeline.get_memory_stats().await
    }

    pub fn evolve(&mut self, proposal: EvolutionProposal) -> Result<String, String> {
        self.evolution_gate.propose_evolution(proposal)
    }

    pub fn evolution_stats(&self) -> HashMap<String, f64> {
        self.evolution_gate.get_evolution_stats()
    }

    /// Expose latest council metrics for external PATSAGi observers
    pub fn get_latest_council_metrics(&self) -> Option<CouncilReadinessMetrics> {
        self.last_council_metrics.clone()
    }
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();
    println!("[Thunder] ONE Organism v14.10 + PATSAGi Council Decision Logic ready");
    organism
}
