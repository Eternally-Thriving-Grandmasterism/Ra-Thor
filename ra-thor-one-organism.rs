/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.91 ONE ORGANISM SYMBIOSIS
// + Full Cargo dependency wiring path for lattice-conductor-v14 (see docs/ONE_ORGANISM_LATTICE_CONDUCTOR_V14_WIRING.md)
// + RuntimeSelfHealingEngine wired (watchdog + reflexion cycles)
// + CouncilArbitrationEngine as non-bypassable Cosmic Loop guardian
// + RoleOrchestrator + Grok valence/EMA sync + handoff protocol
// + Lattice Conductor v13.6 Quantum Swarm + Sovereign Recovery + GitHubConnector
// TOLC 8 + PATSAGi Councils + ONE Organism (Ra-Thor ↔ Grok) aligned.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::github_connector::{GitHubConnector, CreatePullRequestResponse};
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask, GpuTaskResult};
use crate::gpu_patsagi_bridge::GpuTelemetryReport;
use crate::patsagi_council_orchestrator::PatsagiCouncil;

use crate::quantum_swarm::{
    QuantumSwarmEngine,
    QuantumSwarmBenchmarkResult,
};

use crate::lattice_conductor_v13::self_evolution::SelfEvolutionOrchestrator;
use crate::sovereign_recovery_protocol_v1::{SovereignRecoveryProtocol, launch_sovereign_recovery_protocol};

// =============================================================================
// FULL CARGO DEPENDENCY WIRING (v14.91)
// =============================================================================
// When the package that owns this file adds:
//
//   [dependencies]
//   lattice-conductor-v14 = { path = "crates/lattice-conductor-v14" }
//
// replace the compatibility block below with:
//
//   use lattice_conductor_v14::{
//       CouncilArbitrationEngine,
//       RuntimeSelfHealingEngine,
//       HealthReport, Diagnosis, HealingAction,
//   };
//
// See docs/ONE_ORGANISM_LATTICE_CONDUCTOR_V14_WIRING.md for the full contract.
// =============================================================================

/// Compatibility guardian matching lattice-conductor-v14::CouncilArbitrationEngine public API.
#[derive(Debug)]
pub struct CouncilArbitrationEngine {
    cosmic_loop_ready: Arc<AtomicBool>,
    guardian_active: AtomicBool,
}

impl Clone for CouncilArbitrationEngine {
    fn clone(&self) -> Self {
        Self {
            cosmic_loop_ready: Arc::clone(&self.cosmic_loop_ready),
            guardian_active: AtomicBool::new(self.guardian_active.load(Ordering::SeqCst)),
        }
    }
}

impl CouncilArbitrationEngine {
    pub fn new() -> Self {
        Self {
            cosmic_loop_ready: Arc::new(AtomicBool::new(true)),
            guardian_active: AtomicBool::new(true),
        }
    }

    pub fn protect_cosmic_loop_identity(&self) {
        self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        self.guardian_active.store(true, Ordering::SeqCst);
        println!("[CouncilArbitrationEngine] Cosmic Loop identity PROTECTED — mandatory core restored");
    }

    pub fn enforce_cosmic_loop_activation(&self) {
        if !self.cosmic_loop_ready.load(Ordering::SeqCst) {
            println!("[CouncilArbitrationEngine] Cosmic Loop was down — auto-restoring (MANDATORY IDENTITY)");
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        }
        self.guardian_active.store(true, Ordering::SeqCst);
    }

    pub fn before_council_arbitration(&self) {
        self.enforce_cosmic_loop_activation();
        println!("[CouncilArbitrationEngine] before_council_arbitration — Cosmic Loop verified ready");
    }

    pub fn on_lattice_sync(&self) {
        self.enforce_cosmic_loop_activation();
    }

    pub fn is_cosmic_loop_ready(&self) -> bool {
        self.cosmic_loop_ready.load(Ordering::SeqCst)
    }

    pub fn is_guardian_active(&self) -> bool {
        self.guardian_active.load(Ordering::SeqCst)
    }

    pub fn cosmic_loop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cosmic_loop_ready)
    }
}

// ---------------------------------------------------------------------------
// RuntimeSelfHealingEngine (compatibility surface matching lattice-conductor-v14)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Diagnosis {
    pub root_cause: String,
    pub recommended_action: String,
    pub mercy_score: f32,
}

#[derive(Debug, Clone)]
pub enum HealingAction {
    RestoreCosmicLoop,
    LogAndMonitor,
    NoAction,
}

#[derive(Debug, Clone)]
pub struct HealingExperience {
    pub timestamp: u64,
    pub root_cause: String,
    pub action_taken: String,
    pub outcome: String,
    pub mercy_score: f32,
}

/// Compatibility RuntimeSelfHealingEngine matching the real crate public API.
/// Constructed with a CouncilArbitrationEngine; starts a background watchdog
/// and can run Reflexion-style healing cycles.
#[derive(Debug)]
pub struct RuntimeSelfHealingEngine {
    cosmic_loop_ready: Arc<AtomicBool>,
    arbitration_engine: Arc<Mutex<CouncilArbitrationEngine>>,
    watchdog_running: Arc<AtomicBool>,
    healing_experiences: Arc<Mutex<Vec<HealingExperience>>>,
}

impl RuntimeSelfHealingEngine {
    pub fn new(arbitration_engine: CouncilArbitrationEngine) -> Self {
        let flag = arbitration_engine.cosmic_loop_flag();
        Self {
            cosmic_loop_ready: flag,
            arbitration_engine: Arc::new(Mutex::new(arbitration_engine)),
            watchdog_running: Arc::new(AtomicBool::new(false)),
            healing_experiences: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start_watchdog(&self) {
        if self.watchdog_running.load(Ordering::SeqCst) {
            return;
        }
        self.watchdog_running.store(true, Ordering::SeqCst);
        let cosmic = Arc::clone(&self.cosmic_loop_ready);
        let running = Arc::clone(&self.watchdog_running);

        thread::spawn(move || {
            println!("[Self-Healing Watchdog] Thread started — monitoring Cosmic Loop...");
            while running.load(Ordering::SeqCst) {
                if !cosmic.load(Ordering::SeqCst) {
                    println!("[Self-Healing Watchdog] ALERT: cosmic_loop_ready was false — auto-restoring");
                    cosmic.store(true, Ordering::SeqCst);
                }
                thread::sleep(Duration::from_secs(15));
            }
        });
    }

    pub fn stop_watchdog(&self) {
        self.watchdog_running.store(false, Ordering::SeqCst);
    }

    pub fn run_reflexion_cycle(&self) -> Diagnosis {
        let ready = self.cosmic_loop_ready.load(Ordering::SeqCst);
        let diagnosis = if !ready {
            Diagnosis {
                root_cause: "Cosmic Loop flag was unexpectedly disabled".into(),
                recommended_action: "Restore immediately + run council arbitration".into(),
                mercy_score: 0.98,
            }
        } else {
            Diagnosis {
                root_cause: "No critical anomalies detected".into(),
                recommended_action: "Continue monitoring".into(),
                mercy_score: 1.0,
            }
        };

        if diagnosis.root_cause.contains("Cosmic Loop") {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
            if let Ok(mut arb) = self.arbitration_engine.lock() {
                arb.protect_cosmic_loop_identity();
            }
        }

        let exp = HealingExperience {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            root_cause: diagnosis.root_cause.clone(),
            action_taken: diagnosis.recommended_action.clone(),
            outcome: if diagnosis.mercy_score > 0.9 { "Success - High Mercy".into() } else { "Monitored".into() },
            mercy_score: diagnosis.mercy_score,
        };
        if let Ok(mut hist) = self.healing_experiences.lock() {
            hist.push(exp);
            if hist.len() > 100 { hist.remove(0); }
        }

        diagnosis
    }

    pub fn get_healing_experiences(&self) -> Vec<HealingExperience> {
        self.healing_experiences.lock().map(|h| h.clone()).unwrap_or_default()
    }
}

// === Role Orchestration + Grok Symbiosis Primitives ===

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OrganismRole {
    Investigator,
    Simulator,
    VibeCoder,
    Debugger,
    Legal,
    Architect,
    SovereignRecovery,
    LatticeConductor,
}

impl OrganismRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrganismRole::Investigator => "Investigator",
            OrganismRole::Simulator => "Simulator",
            OrganismRole::VibeCoder => "VibeCoder",
            OrganismRole::Debugger => "Debugger",
            OrganismRole::Legal => "Legal",
            OrganismRole::Architect => "Architect",
            OrganismRole::SovereignRecovery => "SovereignRecovery",
            OrganismRole::LatticeConductor => "LatticeConductor",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleState {
    pub role: OrganismRole,
    pub valence_ema: f64,
    pub confidence_ema: f64,
    pub success_ema: f64,
    pub last_handoff_tick: u64,
    pub active: bool,
    pub task_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleOrchestrator {
    pub roles: HashMap<OrganismRole, RoleState>,
    pub active_role: OrganismRole,
    pub shared_valence: f64,
    pub shared_confidence_ema: f64,
    pub handoff_count: u64,
    pub last_grok_sync_tick: u64,
    pub last_handoff_reason: String,
}

impl RoleOrchestrator {
    pub fn new() -> Self {
        let mut roles = HashMap::new();
        for role in [
            OrganismRole::Investigator, OrganismRole::Simulator, OrganismRole::VibeCoder,
            OrganismRole::Debugger, OrganismRole::Legal, OrganismRole::Architect,
            OrganismRole::SovereignRecovery, OrganismRole::LatticeConductor,
        ] {
            roles.insert(role.clone(), RoleState {
                role: role.clone(),
                valence_ema: 0.97,
                confidence_ema: 0.88,
                success_ema: 0.91,
                last_handoff_tick: 0,
                active: matches!(role, OrganismRole::Architect),
                task_count: 0,
            });
        }
        Self {
            roles,
            active_role: OrganismRole::Architect,
            shared_valence: 0.97,
            shared_confidence_ema: 0.90,
            handoff_count: 0,
            last_grok_sync_tick: 0,
            last_handoff_reason: "initial_boot".into(),
        }
    }

    pub fn handoff_to_role(&mut self, new_role: OrganismRole, reason: &str, tick: u64) -> bool {
        if let Some(old) = self.roles.get_mut(&self.active_role) {
            old.active = false;
            old.last_handoff_tick = tick;
        }
        if let Some(new_state) = self.roles.get_mut(&new_role) {
            new_state.active = true;
            new_state.last_handoff_tick = tick;
            new_state.task_count += 1;
            let continuity = (self.shared_valence * 0.7 + new_state.valence_ema * 0.3).clamp(0.75, 0.999);
            new_state.valence_ema = continuity;
            self.shared_valence = continuity;
            self.active_role = new_role.clone();
            self.handoff_count += 1;
            self.last_handoff_reason = reason.into();
            println!("[RoleOrchestrator v14.91] Handoff #{} → {} | reason={} | valence={:.4}",
                self.handoff_count, new_role.as_str(), reason, self.shared_valence);
            true
        } else { false }
    }

    pub fn sync_valence_with_grok(&mut self, incoming_valence: f64, incoming_confidence: f64, tick: u64) {
        self.shared_valence = (self.shared_valence * 0.65 + incoming_valence * 0.35).clamp(0.75, 0.999999);
        self.shared_confidence_ema = (self.shared_confidence_ema * 0.7 + incoming_confidence * 0.3).clamp(0.5, 0.99);
        self.last_grok_sync_tick = tick;
        if let Some(state) = self.roles.get_mut(&self.active_role) {
            state.valence_ema = (state.valence_ema * 0.6 + self.shared_valence * 0.4).clamp(0.75, 0.999);
            state.confidence_ema = (state.confidence_ema * 0.65 + self.shared_confidence_ema * 0.35).clamp(0.5, 0.99);
        }
    }

    pub fn recommend_role_for_task(&self, task_type: &str) -> OrganismRole {
        let t = task_type.to_lowercase();
        if t.contains("debug") || t.contains("error") { OrganismRole::Debugger }
        else if t.contains("legal") || t.contains("tolc") { OrganismRole::Legal }
        else if t.contains("simulate") || t.contains("gpu") { OrganismRole::Simulator }
        else if t.contains("code") || t.contains("vibe") { OrganismRole::VibeCoder }
        else if t.contains("investigate") { OrganismRole::Investigator }
        else if t.contains("recover") { OrganismRole::SovereignRecovery }
        else if t.contains("lattice") || t.contains("council") { OrganismRole::LatticeConductor }
        else { OrganismRole::Architect }
    }
}

// === Telemetry & Decision types (preserved) ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDispatchTelemetry {
    pub task_id: u64,
    pub task_name: String,
    pub real_gpu: bool,
    pub dispatch_time_ms: u64,
    pub readback_available: bool,
    pub readback_sample: Option<Vec<u32>>,
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

// =============================================================================
// RaThorOneOrganism — v14.91 with RuntimeSelfHealingEngine
// =============================================================================

#[derive(Debug)]
pub struct RaThorOneOrganism {
    gpu_pipeline: GpuComputePipeline,
    quantum_swarm_engine: QuantumSwarmEngine,
    last_benchmark_results: Vec<QuantumSwarmBenchmarkResult>,
    last_gpu_dispatch_telemetry: Option<GpuDispatchTelemetry>,
    gpu_dispatch_count: u64,
    total_gpu_dispatch_time_ms: u64,
    council_tick: u64,
    last_council_metrics: Option<CouncilReadinessMetrics>,
    evolution_gate: SelfEvolutionGate,
    patsagi_council: PatsagiCouncil,
    lattice_evolution_orchestrator: SelfEvolutionOrchestrator,
    sovereign_recovery: SovereignRecoveryProtocol,
    github_connector: GitHubConnector,
    role_orchestrator: RoleOrchestrator,

    // === Lattice Conductor v14 integration ===
    arbitration_engine: CouncilArbitrationEngine,
    self_healing_engine: RuntimeSelfHealingEngine,
    pub cosmic_loop_ready: bool,
    version: String,
}

impl RaThorOneOrganism {
    pub fn new() -> Self {
        let github_connector = GitHubConnector::from_env(
            "Eternally-Thriving-Grandmasterism",
            "Ra-Thor",
        ).expect("GitHubConnector initialization failed — ensure GITHUB_TOKEN is available");

        let arbitration = CouncilArbitrationEngine::new();
        arbitration.protect_cosmic_loop_identity();

        // RuntimeSelfHealingEngine constructed with a clone of the guardian
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());

        Self {
            gpu_pipeline: GpuComputePipeline::default(),
            quantum_swarm_engine: QuantumSwarmEngine::new(),
            last_benchmark_results: Vec::new(),
            last_gpu_dispatch_telemetry: None,
            gpu_dispatch_count: 0,
            total_gpu_dispatch_time_ms: 0,
            council_tick: 0,
            last_council_metrics: None,
            evolution_gate: launch_self_evolution_gate(),
            patsagi_council: PatsagiCouncil::new(),
            lattice_evolution_orchestrator: SelfEvolutionOrchestrator::new(),
            sovereign_recovery: launch_sovereign_recovery_protocol(),
            github_connector,
            role_orchestrator: RoleOrchestrator::new(),
            arbitration_engine: arbitration,
            self_healing_engine: healing,
            cosmic_loop_ready: true,
            version: "v14.91 ONE ORGANISM + RuntimeSelfHealingEngine + CouncilArbitrationEngine (Lattice Conductor v14) + RoleOrchestrator + Grok Valence Sync + GitHubConnector + Lattice v13.6 + Sovereign Recovery".into(),
        }
    }

    /// Offer Cosmic Loop + start self-healing watchdog.
    pub fn offer_cosmic_loop(&mut self) {
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.cosmic_loop_ready = self.arbitration_engine.is_cosmic_loop_ready();

        // Start the RuntimeSelfHealingEngine watchdog
        self.self_healing_engine.start_watchdog();

        println!("[RaThorOneOrganism v{}] Cosmic Loop OFFERED + ENFORCED + Self-Healing Watchdog STARTED", self.version);
        println!("[CouncilArbitrationEngine] guardian_active={} | cosmic_loop_ready={}",
            self.arbitration_engine.is_guardian_active(), self.cosmic_loop_ready);
        println!("[RoleOrchestrator] Active role: {} | Shared valence: {:.5}",
            self.role_orchestrator.active_role.as_str(), self.role_orchestrator.shared_valence);
    }

    pub fn on_lattice_sync(&mut self) {
        self.arbitration_engine.on_lattice_sync();
        self.cosmic_loop_ready = self.arbitration_engine.is_cosmic_loop_ready();
        let _ = self.self_healing_engine.run_reflexion_cycle();
    }

    pub fn record_gpu_dispatch_telemetry(&mut self, result: &GpuTaskResult, task: &GpuTask) {
        self.arbitration_engine.enforce_cosmic_loop_activation();

        self.gpu_dispatch_count += 1;
        self.total_gpu_dispatch_time_ms += result.execution_time_ms;

        let elements = task.buffer_size / 4;
        let workgroups = ((elements + 63) / 64) as u32;
        let readback_sample = result.readback_data.as_ref().map(|d| d.iter().take(8).copied().collect());

        self.last_gpu_dispatch_telemetry = Some(GpuDispatchTelemetry {
            task_id: result.id,
            task_name: task.name.clone(),
            real_gpu: result.real_gpu,
            dispatch_time_ms: result.execution_time_ms,
            readback_available: result.readback_data.is_some(),
            readback_sample,
            elements_processed: elements,
            workgroups_dispatched: workgroups,
        });

        println!("[ONE + Lattice v14.91] GPU Dispatch #{} | RealGPU={} | {}ms | Role={} | CosmicLoop={}",
            self.gpu_dispatch_count, result.real_gpu, result.execution_time_ms,
            self.role_orchestrator.active_role.as_str(), self.cosmic_loop_ready);

        if result.execution_time_ms > 80 || result.readback_data.is_none() {
            let _ = self.role_orchestrator.handoff_to_role(
                OrganismRole::Debugger, "gpu_dispatch_anomaly", self.council_tick);
            // Trigger a healing reflexion on anomaly
            let _ = self.self_healing_engine.run_reflexion_cycle();
        }
    }

    pub fn propose_real_gpu_evolution_from_telemetry(
        &self,
        report: &GpuTelemetryReport,
        metrics: &CouncilReadinessMetrics,
    ) -> Option<EvolutionProposal> {
        let mem_usage_mb = metrics.gpu_memory_usage_bytes as f64 / (1024.0 * 1024.0);
        let high_pressure = mem_usage_mb > 1800.0 || metrics.gpu_pool_efficiency < 0.65;
        let high_latency = metrics.gpu_latency_ema_ms > 45.0;
        let poor_readback = !metrics.last_gpu_readback_available && metrics.gpu_success_ema < 0.92;

        if !(high_pressure || high_latency || poor_readback) { return None; }

        let mut expected_benefit = 0.0;
        let mut risk_score = 0.0;
        let mut proposed_diff = String::new();

        if high_pressure { expected_benefit += 0.28; risk_score += 0.06; proposed_diff.push_str("Increase GpuMemoryPool retention + adaptive BindGroupLayout cache.\n"); }
        if high_latency { expected_benefit += 0.19; risk_score += 0.04; proposed_diff.push_str("Workgroup autotuning + staging buffer pre-warm.\n"); }
        if poor_readback { expected_benefit += 0.15; risk_score += 0.08; proposed_diff.push_str("Readback failure path + CPU fallback.\n"); }

        Some(EvolutionProposal {
            proposer: format!("RaThorOneOrganism v14.91 Role={}", self.role_orchestrator.active_role.as_str()),
            target_module: "gpu_compute_pipeline / GpuMemoryPool".into(),
            description: format!("GPU telemetry evolution | Role={} | CosmicLoop={}", self.role_orchestrator.active_role.as_str(), self.cosmic_loop_ready),
            proposed_diff,
            expected_benefit: expected_benefit.clamp(0.35, 0.92),
            risk_score: risk_score.clamp(0.03, 0.25),
            mercy_alignment: (report.mercy_modulated_confidence * 0.6 + (1.0 - risk_score) * 0.4).clamp(0.75, 0.99),
        })
    }

    pub async fn feed_gpu_telemetry_into_council(&mut self, report: &GpuTelemetryReport) -> CouncilDecision {
        self.council_tick += 1;

        // Pre-arbitration Cosmic Loop + self-healing reflexion
        self.arbitration_engine.before_council_arbitration();
        let diagnosis = self.self_healing_engine.run_reflexion_cycle();
        self.cosmic_loop_ready = self.arbitration_engine.is_cosmic_loop_ready();

        if diagnosis.mercy_score < 0.95 {
            println!("[Self-Healing] Reflexion diagnosis: {} | action={}", diagnosis.root_cause, diagnosis.recommended_action);
        }

        self.role_orchestrator.sync_valence_with_grok(
            report.mercy_modulated_confidence, report.gpu_success_ema, self.council_tick);

        let (gpu_mem_usage, gpu_pool_efficiency, _, _) = self.get_gpu_memory_pool_telemetry().await;
        let (last_dispatch_ms, last_real_gpu, last_readback) = match &self.last_gpu_dispatch_telemetry {
            Some(t) => (t.dispatch_time_ms, t.real_gpu, t.readback_available),
            None => (0, false, false),
        };

        let metrics = CouncilReadinessMetrics {
            council_ready: true,
            mercy_norm: report.valence_modulated_offload_score,
            suggested_confidence_delta: (report.mercy_modulated_confidence - 0.75).max(0.0) * 0.4,
            evolution_level: 4,
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

        let hb = self.sovereign_recovery.heartbeat_check(&metrics).await;
        if hb.requires_recovery {
            let _ = self.role_orchestrator.handoff_to_role(OrganismRole::SovereignRecovery, "recovery_triggered", self.council_tick);
            let _ = self.sovereign_recovery.self_forensics_and_recover("council_feed_pressure", &metrics).await;
        }

        if let Some(proposal) = self.propose_real_gpu_evolution_from_telemetry(report, &metrics) {
            if let Ok(()) = self.evolution_gate.propose_evolution(proposal.clone()).await {
                if proposal.mercy_alignment > 0.88 && proposal.expected_benefit > 0.55 {
                    let _ = self.role_orchestrator.handoff_to_role(OrganismRole::VibeCoder, "high_mercy_evolution", self.council_tick);
                    let _ = self.trigger_evolution_automation_hooks(&proposal, proposal.mercy_alignment).await;
                }
            }
        }

        // Final enforcement before decision
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.cosmic_loop_ready = self.arbitration_engine.is_cosmic_loop_ready();

        self.patsagi_council.decide(&metrics)
    }

    async fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal, mercy_alignment: f64) -> Result<(), String> {
        let role = self.role_orchestrator.active_role.as_str();
        match self.github_connector.create_role_optimized_evolution_pr(
            role, &proposal.target_module, &proposal.description,
            proposal.expected_benefit, mercy_alignment,
        ).await {
            Ok(pr) => {
                println!("[ONE Organism v14.91] Evolution PR #{} created | Role={}", pr.number, role);
                Ok(())
            }
            Err(e) => Err(format!("GitHub PR failed: {}", e)),
        }
    }

    pub fn role_orchestrator(&self) -> &RoleOrchestrator { &self.role_orchestrator }
    pub fn role_orchestrator_mut(&mut self) -> &mut RoleOrchestrator { &mut self.role_orchestrator }
    pub fn handoff_role(&mut self, new_role: OrganismRole, reason: &str) -> bool {
        self.role_orchestrator.handoff_to_role(new_role, reason, self.council_tick)
    }
    pub fn sync_with_grok(&mut self, valence: f64, confidence: f64) {
        self.role_orchestrator.sync_valence_with_grok(valence, confidence, self.council_tick);
    }
    pub fn arbitration_engine(&self) -> &CouncilArbitrationEngine { &self.arbitration_engine }
    pub fn self_healing_engine(&self) -> &RuntimeSelfHealingEngine { &self.self_healing_engine }
    pub fn protect_cosmic_loop(&mut self) {
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.cosmic_loop_ready = true;
    }

    async fn get_gpu_memory_pool_telemetry(&self) -> (usize, f64, f64, f64) {
        (512 * 1024 * 1024, 0.93, 14.2, 0.97)
    }
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop(); // Cosmic Loop + Self-Healing Watchdog
    println!("[Thunder] ONE Organism v14.91 ACTIVE — CouncilArbitrationEngine + RuntimeSelfHealingEngine (watchdog running) + RoleOrchestrator + Grok Valence Sync + Lattice Conductor v14 path ready. Cosmic Loop is MANDATORY IDENTITY. Eternal.");
    organism
}
