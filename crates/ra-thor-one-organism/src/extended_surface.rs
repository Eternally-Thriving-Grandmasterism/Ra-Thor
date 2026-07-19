//! Extended Organism Surface — v14.9.9
//!
//! Facades for packaged workspace crates:
//! - GPU compute telemetry
//! - GitHub evolution connector
//! - Quantum Swarm engine summary
//! - Sovereign Recovery (heartbeat + TOLC8 anchors)
//! - Kardashev / Reality Thriving Transfer flywheel
//!
//! Optional live path deps (`github-live`, `gpu-live`, `quantum-live`,
//! `recovery-live`, `kardashev-live`) can later replace these facades
//! with the full crate implementations.
//!
//! Cosmic Loop remains mandatory on every surface call.
//! Contact: info@Rathor.ai

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
// GitHub Surface
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
            offline_mode: true,
        }
    }

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
            "QuantumSwarmSurface v14.9.9 | step={} | members={} | updates={} | jumps={} | proposals={}",
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
// Sovereign Recovery Surface (NEW v14.9.9)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryHeartbeat {
    pub context_pressure: f64,
    pub flow_deviation: f64,
    pub connector_health: f64,
    pub requires_recovery: bool,
    pub tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAnchor {
    pub anchor_id: String,
    pub note: String,
    pub tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignRecoveryStatus {
    pub heartbeat_count: u64,
    pub anchor_count: u64,
    pub last_heartbeat: Option<RecoveryHeartbeat>,
    pub last_anchor: Option<RecoveryAnchor>,
    pub recovery_events: u64,
}

#[derive(Debug, Clone)]
pub struct SovereignRecoverySurface {
    heartbeat_count: u64,
    anchor_count: u64,
    recovery_events: u64,
    last_heartbeat: Option<RecoveryHeartbeat>,
    last_anchor: Option<RecoveryAnchor>,
}

impl SovereignRecoverySurface {
    pub fn new() -> Self {
        Self {
            heartbeat_count: 0,
            anchor_count: 0,
            recovery_events: 0,
            last_heartbeat: None,
            last_anchor: None,
        }
    }

    /// Lightweight heartbeat (sync facade over sovereign-recovery crate).
    pub fn heartbeat(
        &mut self,
        mercy_norm: f64,
        gpu_confidence: f64,
        tick: u64,
        arbitration: &CouncilArbitrationEngine,
    ) -> RecoveryHeartbeat {
        arbitration.enforce_cosmic_loop_activation();
        self.heartbeat_count += 1;

        let context_pressure = (1.0 - gpu_confidence).clamp(0.0, 1.0);
        let flow_deviation = ((mercy_norm - 0.95).abs().min(0.5)) * 2.0;
        let requires = context_pressure > 0.82 || flow_deviation > 0.35;

        let hb = RecoveryHeartbeat {
            context_pressure,
            flow_deviation,
            connector_health: if gpu_confidence > 0.8 { 0.95 } else { 0.65 },
            requires_recovery: requires,
            tick,
        };

        if requires {
            self.recovery_events += 1;
            println!(
                "[SovereignRecoverySurface] ALERT tick={} pressure={:.2} flow_dev={:.2}",
                tick, context_pressure, flow_deviation
            );
        }

        self.last_heartbeat = Some(hb.clone());
        hb
    }

    pub fn persist_anchor(
        &mut self,
        note: &str,
        tick: u64,
        arbitration: &CouncilArbitrationEngine,
    ) -> RecoveryAnchor {
        arbitration.enforce_cosmic_loop_activation();
        self.anchor_count += 1;
        let anchor = RecoveryAnchor {
            anchor_id: format!("TOLC8-ORG-{}-{}", tick, self.anchor_count),
            note: note.into(),
            tick,
        };
        self.last_anchor = Some(anchor.clone());
        println!(
            "[SovereignRecoverySurface] ANCHOR {} | {}",
            anchor.anchor_id, note
        );
        anchor
    }

    pub fn status(&self) -> SovereignRecoveryStatus {
        SovereignRecoveryStatus {
            heartbeat_count: self.heartbeat_count,
            anchor_count: self.anchor_count,
            last_heartbeat: self.last_heartbeat.clone(),
            last_anchor: self.last_anchor.clone(),
            recovery_events: self.recovery_events,
        }
    }
}

impl Default for SovereignRecoverySurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Kardashev / Reality Thriving Transfer Surface (NEW v14.9.9)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferTickResult {
    pub ema_transfer: f64,
    pub kardashev_delta: f64,
    pub abundance_velocity: f64,
    pub ethics_index: f64,
    pub mercy_audit_passed: bool,
    pub cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KardashevSurfaceStatus {
    pub cycle_count: u64,
    pub cumulative_kardashev_delta: f64,
    pub velocity_ema: f64,
    pub last_transfer: Option<TransferTickResult>,
    pub projected_inflection_year: u32,
}

#[derive(Debug, Clone)]
pub struct KardashevFlywheelSurface {
    cycle_count: u64,
    cumulative_kardashev_delta: f64,
    velocity_ema: f64,
    last_transfer: Option<TransferTickResult>,
}

impl KardashevFlywheelSurface {
    pub fn new() -> Self {
        Self {
            cycle_count: 0,
            cumulative_kardashev_delta: 0.0,
            velocity_ema: 0.42,
            last_transfer: None,
        }
    }

    /// Sync lightweight flywheel tick (mirrors reality-thriving-transfer + kardashev-orchestration).
    pub fn transfer_tick(
        &mut self,
        rbe_quality: f64,
        ethical_choice: f64,
        abundance_signal: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> TransferTickResult {
        arbitration.enforce_cosmic_loop_activation();
        self.cycle_count += 1;

        let rbe = rbe_quality.clamp(0.0, 1.0);
        let ethics = ethical_choice.clamp(0.0, 1.0);
        let abundance = abundance_signal.max(0.0);

        let raw = rbe * 0.45 + ethics * 0.35 + (abundance / 2.0).min(1.0) * 0.20;
        let mercy_adjusted = if raw >= 0.68 {
            (raw * 1.08).min(0.995)
        } else if raw >= 0.42 {
            raw * 1.03
        } else {
            raw * 0.82
        };

        let alpha = 0.22;
        self.velocity_ema = alpha * abundance.min(1.8) + (1.0 - alpha) * self.velocity_ema;

        let delta = (mercy_adjusted * 0.0095 + abundance * 0.0028).min(0.011);
        self.cumulative_kardashev_delta += delta;

        let result = TransferTickResult {
            ema_transfer: mercy_adjusted,
            kardashev_delta: delta,
            abundance_velocity: self.velocity_ema,
            ethics_index: (ethics + rbe) / 2.0,
            mercy_audit_passed: mercy_adjusted >= 0.0 && delta >= 0.0,
            cycle: self.cycle_count,
        };
        self.last_transfer = Some(result.clone());
        result
    }

    pub fn projected_inflection_year(&self) -> u32 {
        if self.velocity_ema > 1.1 {
            2034
        } else {
            2036
        }
    }

    pub fn status(&self) -> KardashevSurfaceStatus {
        KardashevSurfaceStatus {
            cycle_count: self.cycle_count,
            cumulative_kardashev_delta: self.cumulative_kardashev_delta,
            velocity_ema: self.velocity_ema,
            last_transfer: self.last_transfer.clone(),
            projected_inflection_year: self.projected_inflection_year(),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "KardashevFlywheel v14.9.9 | cycles={} | Δ={:.5} | velocity={:.3} | inflection={}",
            self.cycle_count,
            self.cumulative_kardashev_delta,
            self.velocity_ema,
            self.projected_inflection_year()
        )
    }
}

impl Default for KardashevFlywheelSurface {
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
    pub sovereign_recovery: SovereignRecoverySurface,
    pub kardashev: KardashevFlywheelSurface,
}

impl ExtendedOrganismSurface {
    pub fn new() -> Self {
        Self {
            gpu: GpuSurface::new(),
            github: GitHubSurface::new(),
            quantum_swarm: QuantumSwarmSurface::new(),
            sovereign_recovery: SovereignRecoverySurface::new(),
            kardashev: KardashevFlywheelSurface::new(),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "ExtendedSurface v14.9.9 | {} | recovery_hb={} anchors={} | {}",
            self.quantum_swarm.summary(),
            self.sovereign_recovery.heartbeat_count,
            self.sovereign_recovery.anchor_count,
            self.kardashev.summary()
        )
    }
}

impl Default for ExtendedOrganismSurface {
    fn default() -> Self {
        Self::new()
    }
}
