//! Extended Organism Surface — v14.9.2
//!
//! Lightweight, dependency-free facades for the three historical root modules:
//! - GPU compute telemetry
//! - GitHub evolution connector
//! - Quantum Swarm engine summary
//!
//! Full production implementations remain at repo root
//! (`gpu_compute_pipeline.rs`, `github_connector.rs`, `quantum_swarm.rs`).
//! This module gives `OneOrganismCore` a clean, compilable extended API
//! that can later path-depend on those crates when they are packaged.
//!
//! Cosmic Loop remains mandatory on every surface call.

use serde::{Deserialize, Serialize};

use crate::CouncilArbitrationEngine;

// =============================================================================
// GPU Surface
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDispatchTelemetry {
    pub task_id: u64,
    pub task_name: String,
    pub real_gpu: bool,
    pub dispatch_time_ms: u64,
    pub readback_available: bool,
    pub elements_processed: usize,
    pub workgroups_dispatched: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSurfaceStatus {
    pub dispatch_count: u64,
    pub total_dispatch_time_ms: u64,
    pub last_telemetry: Option<GpuDispatchTelemetry>,
    pub pool_efficiency: f64,
    pub memory_usage_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GpuSurface {
    dispatch_count: u64,
    total_dispatch_time_ms: u64,
    last_telemetry: Option<GpuDispatchTelemetry>,
    pool_efficiency: f64,
    memory_usage_bytes: usize,
}

impl GpuSurface {
    pub fn new() -> Self {
        Self {
            dispatch_count: 0,
            total_dispatch_time_ms: 0,
            last_telemetry: None,
            pool_efficiency: 0.93,
            memory_usage_bytes: 512 * 1024 * 1024,
        }
    }

    /// Record a dispatch (real or simulated). Enforces Cosmic Loop via arbitration.
    pub fn record_dispatch(
        &mut self,
        task_name: &str,
        dispatch_time_ms: u64,
        real_gpu: bool,
        elements: usize,
        arbitration: &CouncilArbitrationEngine,
    ) -> GpuDispatchTelemetry {
        arbitration.enforce_cosmic_loop_activation();

        self.dispatch_count += 1;
        self.total_dispatch_time_ms += dispatch_time_ms;

        let workgroups = ((elements + 63) / 64) as u32;
        let tel = GpuDispatchTelemetry {
            task_id: self.dispatch_count,
            task_name: task_name.into(),
            real_gpu,
            dispatch_time_ms,
            readback_available: true,
            elements_processed: elements,
            workgroups_dispatched: workgroups,
        };
        self.last_telemetry = Some(tel.clone());

        if dispatch_time_ms > 80 {
            println!(
                "[GpuSurface] anomaly: {}ms on {} — Debugger handoff recommended",
                dispatch_time_ms, task_name
            );
        }

        tel
    }

    pub fn status(&self) -> GpuSurfaceStatus {
        GpuSurfaceStatus {
            dispatch_count: self.dispatch_count,
            total_dispatch_time_ms: self.total_dispatch_time_ms,
            last_telemetry: self.last_telemetry.clone(),
            pool_efficiency: self.pool_efficiency,
            memory_usage_bytes: self.memory_usage_bytes,
        }
    }
}

impl Default for GpuSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GitHub Surface (offline-capable)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPrIntent {
    pub role: String,
    pub target_module: String,
    pub description: String,
    pub expected_benefit: f64,
    pub mercy_alignment: f64,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubSurfaceStatus {
    pub intended_prs: usize,
    pub last_intent: Option<EvolutionPrIntent>,
    pub offline_mode: bool,
}

#[derive(Debug, Clone)]
pub struct GitHubSurface {
    intended_prs: Vec<EvolutionPrIntent>,
    offline_mode: bool,
}

impl GitHubSurface {
    pub fn new() -> Self {
        Self {
            intended_prs: Vec::new(),
            offline_mode: true, // no network by default; safe for core crate
        }
    }

    /// Queue an evolution PR intent (does not hit the network).
    /// Real `github_connector.rs` can drain this queue when wired.
    pub fn queue_evolution_pr(
        &mut self,
        role: &str,
        target_module: &str,
        description: &str,
        expected_benefit: f64,
        mercy_alignment: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> EvolutionPrIntent {
        arbitration.enforce_cosmic_loop_activation();
        arbitration.before_council_arbitration();

        let title = format!(
            "[ONE Organism] {} evolution: {} (benefit={:.2}, mercy={:.3})",
            role, target_module, expected_benefit, mercy_alignment
        );
        let intent = EvolutionPrIntent {
            role: role.into(),
            target_module: target_module.into(),
            description: description.into(),
            expected_benefit,
            mercy_alignment,
            title: title.clone(),
        };
        self.intended_prs.push(intent.clone());
        println!(
            "[GitHubSurface] queued PR intent #{} | {} | offline={}",
            self.intended_prs.len(),
            title,
            self.offline_mode
        );
        intent
    }

    pub fn drain_intents(&mut self) -> Vec<EvolutionPrIntent> {
        std::mem::take(&mut self.intended_prs)
    }

    pub fn status(&self) -> GitHubSurfaceStatus {
        GitHubSurfaceStatus {
            intended_prs: self.intended_prs.len(),
            last_intent: self.intended_prs.last().cloned(),
            offline_mode: self.offline_mode,
        }
    }
}

impl Default for GitHubSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Quantum Swarm Surface
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmConfig {
    pub gaussian_scale: f64,
    pub mean_best_influence: f64,
    pub entanglement_modulation: f64,
    pub quantum_jump_base_prob: f64,
}

impl Default for QuantumSwarmConfig {
    fn default() -> Self {
        Self {
            gaussian_scale: 0.15,
            mean_best_influence: 0.35,
            entanglement_modulation: 0.25,
            quantum_jump_base_prob: 0.08,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmStatus {
    pub step: u64,
    pub member_count: usize,
    pub total_weight_updates: u64,
    pub total_adaptive_jumps: u64,
    pub total_proposals: u64,
    pub config: QuantumSwarmConfig,
}

#[derive(Debug, Clone)]
pub struct QuantumSwarmSurface {
    config: QuantumSwarmConfig,
    step: u64,
    member_count: usize,
    total_weight_updates: u64,
    total_adaptive_jumps: u64,
    total_proposals: u64,
}

impl QuantumSwarmSurface {
    pub fn new() -> Self {
        Self {
            config: QuantumSwarmConfig::default(),
            step: 0,
            member_count: 0,
            total_weight_updates: 0,
            total_adaptive_jumps: 0,
            total_proposals: 0,
        }
    }

    pub fn register_members(&mut self, count: usize) {
        self.member_count = count;
    }

    /// Lightweight evolution tick (no rand / GPU). Cosmic Loop enforced.
    pub fn evolution_tick(
        &mut self,
        severity: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> f64 {
        arbitration.enforce_cosmic_loop_activation();
        self.step += 1;
        self.total_weight_updates += 1;

        if severity >= 0.35 {
            self.total_adaptive_jumps += 1;
        }
        if severity >= 0.15 {
            self.total_proposals += 1;
        }

        // synthetic quantum ratio
        (self.config.gaussian_scale * (1.0 + severity)).min(1.0)
    }

    pub fn status(&self) -> QuantumSwarmStatus {
        QuantumSwarmStatus {
            step: self.step,
            member_count: self.member_count,
            total_weight_updates: self.total_weight_updates,
            total_adaptive_jumps: self.total_adaptive_jumps,
            total_proposals: self.total_proposals,
            config: self.config.clone(),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "QuantumSwarmSurface v14.9.2 | step={} | members={} | updates={} | jumps={} | proposals={}",
            self.step,
            self.member_count,
            self.total_weight_updates,
            self.total_adaptive_jumps,
            self.total_proposals
        )
    }
}

impl Default for QuantumSwarmSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Aggregated Extended Surface
// =============================================================================

#[derive(Debug, Clone)]
pub struct ExtendedOrganismSurface {
    pub gpu: GpuSurface,
    pub github: GitHubSurface,
    pub quantum_swarm: QuantumSwarmSurface,
}

impl ExtendedOrganismSurface {
    pub fn new() -> Self {
        Self {
            gpu: GpuSurface::new(),
            github: GitHubSurface::new(),
            quantum_swarm: QuantumSwarmSurface::new(),
        }
    }
}

impl Default for ExtendedOrganismSurface {
    fn default() -> Self {
        Self::new()
    }
}
