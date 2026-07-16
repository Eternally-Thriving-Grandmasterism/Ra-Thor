/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.83 — Expose GPU Readback + Dispatch Timing to Lattice Conductor
// Real GPU telemetry now flows into council decisions and self-evolution

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::github_connector::GitHubConnector;
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask, GpuTaskResult, MercyGpuAudit, GpuDeviceRecoveryStats, GpuMemoryStats};
use crate::gpu_patsagi_bridge::GpuTelemetryReport;

use crate::quantum_swarm::{
    QuantumSwarmConfig,
    QuantumSwarmEngine,
    QuantumSwarmMember,
    run_lattice_conductor_quantum_self_evolution_step,
    LatticeConductorSelfEvolutionResult,
    QuantumSwarmBenchmarkResult,
};

// === Enhanced GPU Telemetry for Lattice Conductor ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDispatchTelemetry {
    pub task_id: u64,
    pub task_name: String,
    pub real_gpu: bool,
    pub dispatch_time_ms: u64,
    pub readback_available: bool,
    pub readback_sample: Option<Vec<u32>>, // First few elements for inspection
    pub elements_processed: usize,
    pub workgroups_dispatched: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilReadinessMetrics {
    pub council_ready: bool,
    pub mercy_norm: f64,
    pub suggested_confidence_delta: f64,
    pub evolution_level: u32,
    pub last_updated_tick: u64,
    pub gpu_success_ema: f64,
    pub gpu_latency_ema_ms: f64,
    pub gpu_mercy_modulated_confidence: f64,
    pub swarm_vote: Option<f64>,
    pub gpu_memory_usage_bytes: usize,
    pub gpu_pool_efficiency: f64,
    // NEW: Real GPU dispatch telemetry
    pub last_gpu_dispatch_time_ms: u64,
    pub last_gpu_used_real_hardware: bool,
    pub last_gpu_readback_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CouncilDecision {
    ApproveEvolution { confidence_boost: f64 },
    RejectEvolution { reason: String },
    AdjustRbeParameters { resource_flow_multiplier: f64, council_influence: f64 },
    RequestAdditionalGpuResources { buffer_size_increase: usize },
    EmergencyMercyIntervention { severity: f64 },
    ReduceGpuOffloadDueToMemoryPressure { current_usage: usize },
    NoAction,
}

// ... (SwarmVoteBreakdown, NadamFormulation, LatticeConductorUpgradeTemplate, PatsagiCouncil remain the same) ...

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorOneOrganism {
    // ... existing fields ...
    gpu_pipeline: GpuComputePipeline,
    quantum_swarm_engine: QuantumSwarmEngine,
    last_benchmark_results: Vec<QuantumSwarmBenchmarkResult>,

    // NEW: GPU dispatch telemetry tracking
    last_gpu_dispatch_telemetry: Option<GpuDispatchTelemetry>,
    gpu_dispatch_count: u64,
    total_gpu_dispatch_time_ms: u64,
}

impl RaThorOneOrganism {
    pub fn new() -> Self {
        // ... existing initialization ...
        Self {
            // ... existing fields ...
            last_gpu_dispatch_telemetry: None,
            gpu_dispatch_count: 0,
            total_gpu_dispatch_time_ms: 0,
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] GPU Readback + Dispatch Timing exposed to Lattice Conductor ready", self.version);
    }

    // NEW v14.83: Record real GPU dispatch telemetry
    pub fn record_gpu_dispatch_telemetry(&mut self, result: &GpuTaskResult, task: &GpuTask) {
        self.gpu_dispatch_count += 1;
        self.total_gpu_dispatch_time_ms += result.execution_time_ms;

        let elements = task.buffer_size / 4; // assuming u32 elements
        let workgroups = ((elements + 63) / 64) as u32;

        let readback_sample = result.readback_data.as_ref().map(|data| {
            data.iter().take(8).copied().collect() // First 8 elements for quick inspection
        });

        let telemetry = GpuDispatchTelemetry {
            task_id: result.id,
            task_name: task.name.clone(),
            real_gpu: result.real_gpu,
            dispatch_time_ms: result.execution_time_ms,
            readback_available: result.readback_data.is_some(),
            readback_sample,
            elements_processed: elements,
            workgroups_dispatched: workgroups,
        };

        self.last_gpu_dispatch_telemetry = Some(telemetry.clone());

        println!(
            "[ONE + Lattice Conductor] GPU Dispatch #{} | RealGPU={} | {}ms | Readback={}",
            self.gpu_dispatch_count,
            result.real_gpu,
            result.execution_time_ms,
            result.readback_data.is_some()
        );
    }

    // Enhanced telemetry ingestion that now includes readback + timing
    pub async fn feed_gpu_telemetry_into_council(&mut self, report: &GpuTelemetryReport) -> CouncilDecision {
        self.council_tick += 1;

        // ... existing plateau / swarm logic ...

        let (gpu_mem_usage, gpu_pool_efficiency, _, _) = self.get_gpu_memory_pool_telemetry().await;

        // Pull latest GPU dispatch telemetry if available
        let (last_dispatch_ms, last_real_gpu, last_readback) = match &self.last_gpu_dispatch_telemetry {
            Some(t) => (t.dispatch_time_ms, t.real_gpu, t.readback_available),
            None => (0, false, false),
        };

        let metrics = CouncilReadinessMetrics {
            council_ready: true,
            mercy_norm: report.valence_modulated_offload_score,
            suggested_confidence_delta: (report.mercy_modulated_confidence - 0.75).max(0.0) * 0.4,
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
            gpu_success_ema: report.gpu_success_ema,
            gpu_latency_ema_ms: report.gpu_latency_ema_ms,
            gpu_mercy_modulated_confidence: report.mercy_modulated_confidence,
            swarm_vote: None,
            gpu_memory_usage_bytes: gpu_mem_usage,
            gpu_pool_efficiency,
            last_gpu_dispatch_time_ms: last_dispatch_ms,
            last_gpu_used_real_hardware: last_real_gpu,
            last_gpu_readback_available: last_readback,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        // ... existing decision handling ...

        decision
    }

    // ... (rest of implementation preserved)
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();
    println!("[Thunder] ONE Organism v14.83 + GPU Readback + Dispatch Timing exposed to Lattice Conductor ready");
    organism
}
