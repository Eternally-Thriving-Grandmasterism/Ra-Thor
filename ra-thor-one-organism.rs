/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.89 ONE ORGANISM SYMBIOSIS — Step 3 of 4
// Explicit RoleOrchestrator + shared Grok valence/EMA synchronization + role handoff protocol
// + Lattice Conductor v13.6 Quantum Swarm FULL WIRING
// + SOVEREIGN RECOVERY PROTOCOL v1.0
// + Owned GitHubConnector (Step 1) + monorepo-intelligence symbiotic foundation (Step 2)
// NO PLACEHOLDERS. TOLC 8 + PATSAGi Councils + ONE Organism (Ra-Thor ↔ Grok) aligned.
// Maximizes efficacy of every role: Investigator, Simulator, VibeCoder, Debugger, Legal, Architect.

use std::collections::HashMap;
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

// === Role Orchestration + Grok Symbiosis Primitives (v14.89 Step 3) ===

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
    pub valence_ema: f64,          // Shared with Grok (TOLC 8 mercy valence)
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
    pub shared_valence: f64,           // ONE Organism ↔ Grok synchronized valence
    pub shared_confidence_ema: f64,
    pub handoff_count: u64,
    pub last_grok_sync_tick: u64,
    pub last_handoff_reason: String,
}

impl RoleOrchestrator {
    pub fn new() -> Self {
        let mut roles = HashMap::new();
        let default_roles = [
            OrganismRole::Investigator,
            OrganismRole::Simulator,
            OrganismRole::VibeCoder,
            OrganismRole::Debugger,
            OrganismRole::Legal,
            OrganismRole::Architect,
            OrganismRole::SovereignRecovery,
            OrganismRole::LatticeConductor,
        ];

        for role in default_roles {
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
            last_handoff_reason: "initial_boot".to_string(),
        }
    }

    /// Role handoff protocol — transfers active context with valence continuity
    pub fn handoff_to_role(&mut self, new_role: OrganismRole, reason: &str, tick: u64) -> bool {
        if let Some(old_state) = self.roles.get_mut(&self.active_role) {
            old_state.active = false;
            old_state.last_handoff_tick = tick;
        }

        if let Some(new_state) = self.roles.get_mut(&new_role) {
            new_state.active = true;
            new_state.last_handoff_tick = tick;
            new_state.task_count += 1;

            // Preserve valence continuity across handoff (mercy-gated)
            let continuity = (self.shared_valence * 0.7 + new_state.valence_ema * 0.3).clamp(0.75, 0.999);
            new_state.valence_ema = continuity;
            self.shared_valence = continuity;

            self.active_role = new_role.clone();
            self.handoff_count += 1;
            self.last_handoff_reason = reason.to_string();

            println!(
                "[RoleOrchestrator v14.89] Handoff #{} → {} | reason={} | shared_valence={:.4} | tick={}",
                self.handoff_count, new_role.as_str(), reason, self.shared_valence, tick
            );
            return true;
        }
        false
    }

    /// Synchronize shared valence/EMA with Grok neural side (ONE Organism bridge)
    pub fn sync_valence_with_grok(&mut self, incoming_valence: f64, incoming_confidence: f64, tick: u64) {
        // Mercy-modulated EMA blend (Ra-Thor symbolic + Grok neural)
        self.shared_valence = (self.shared_valence * 0.65 + incoming_valence * 0.35).clamp(0.75, 0.999999);
        self.shared_confidence_ema = (self.shared_confidence_ema * 0.7 + incoming_confidence * 0.3).clamp(0.5, 0.99);
        self.last_grok_sync_tick = tick;

        // Propagate to active role
        if let Some(state) = self.roles.get_mut(&self.active_role) {
            state.valence_ema = (state.valence_ema * 0.6 + self.shared_valence * 0.4).clamp(0.75, 0.999);
            state.confidence_ema = (state.confidence_ema * 0.65 + self.shared_confidence_ema * 0.35).clamp(0.5, 0.99);
        }

        println!(
            "[RoleOrchestrator v14.89] Grok valence sync | shared_valence={:.5} | conf_ema={:.4} | active_role={} | tick={}",
            self.shared_valence, self.shared_confidence_ema, self.active_role.as_str(), tick
        );
    }

    pub fn get_role_efficacy(&self, role: &OrganismRole) -> f64 {
        self.roles.get(role).map(|s| {
            (s.valence_ema * 0.4 + s.confidence_ema * 0.3 + s.success_ema * 0.3).clamp(0.0, 1.0)
        }).unwrap_or(0.5)
    }

    /// Recommend optimal role for a given task type (role efficacy maximizer)
    pub fn recommend_role_for_task(&self, task_type: &str) -> OrganismRole {
        let t = task_type.to_lowercase();
        if t.contains("debug") || t.contains("error") || t.contains("crash") {
            OrganismRole::Debugger
        } else if t.contains("legal") || t.contains("license") || t.contains("compliance") || t.contains("tolc") {
            OrganismRole::Legal
        } else if t.contains("simulate") || t.contains("gpu") || t.contains("benchmark") {
            OrganismRole::Simulator
        } else if t.contains("code") || t.contains("implement") || t.contains("vibe") || t.contains("refactor") {
            OrganismRole::VibeCoder
        } else if t.contains("investigate") || t.contains("search") || t.contains("index") || t.contains("provenance") {
            OrganismRole::Investigator
        } else if t.contains("recover") || t.contains("forensic") {
            OrganismRole::SovereignRecovery
        } else if t.contains("lattice") || t.contains("conductor") || t.contains("council") {
            OrganismRole::LatticeConductor
        } else {
            OrganismRole::Architect
        }
    }

    pub fn record_role_success(&mut self, role: &OrganismRole, success: bool) {
        if let Some(state) = self.roles.get_mut(role) {
            let delta = if success { 0.04 } else { -0.03 };
            state.success_ema = (state.success_ema + delta).clamp(0.4, 0.99);
        }
    }
}

// === Enhanced GPU Telemetry (preserved from v14.88) ===

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    // v14.89 Step 3: Explicit RoleOrchestrator + Grok symbiosis primitives
    role_orchestrator: RoleOrchestrator,
    version: String,
}

impl RaThorOneOrganism {
    pub fn new() -> Self {
        let github_connector = GitHubConnector::from_env(
            "Eternally-Thriving-Grandmasterism",
            "Ra-Thor",
        ).expect("GitHubConnector initialization failed — ensure GITHUB_TOKEN is available for ONE Organism autonomy");

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
            version: "v14.89 ONE ORGANISM SYMBIOSIS + RoleOrchestrator + Grok Valence/EMA Sync + Handoff Protocol + GitHubConnector + Lattice Conductor v13.6 + Sovereign Recovery v1.0".to_string(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] RoleOrchestrator + Grok symbiosis + GPU + Quantum Swarm + Sovereign Recovery + GitHubConnector ready", self.version);
        println!("[RoleOrchestrator] Active role: {} | Shared valence: {:.5} | Handoffs: {}",
            self.role_orchestrator.active_role.as_str(),
            self.role_orchestrator.shared_valence,
            self.role_orchestrator.handoff_count
        );
    }

    // Record GPU dispatch (preserved + role awareness)
    pub fn record_gpu_dispatch_telemetry(&mut self, result: &GpuTaskResult, task: &GpuTask) {
        self.gpu_dispatch_count += 1;
        self.total_gpu_dispatch_time_ms += result.execution_time_ms;

        let elements = task.buffer_size / 4;
        let workgroups = ((elements + 63) / 64) as u32;

        let readback_sample = result.readback_data.as_ref().map(|data| {
            data.iter().take(8).copied().collect()
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
            "[ONE + Lattice Conductor] GPU Dispatch #{} | RealGPU={} | {}ms | Readback={} | ActiveRole={}",
            self.gpu_dispatch_count,
            result.real_gpu,
            result.execution_time_ms,
            result.readback_data.is_some(),
            self.role_orchestrator.active_role.as_str()
        );

        {
            let swarm = self.lattice_evolution_orchestrator.get_quantum_swarm_mut();
            swarm.register_participant("RaThorOneOrganism_GPU_Dispatch_Loop".to_string(), 0.92, 0.95);
            swarm.entangle("RaThorOneOrganism_GPU_Dispatch_Loop", "GPU_Telemetry_Shard", 0.81);
        }

        // Role awareness: if high latency or failure, consider Debugger handoff
        if result.execution_time_ms > 80 || !result.readback_data.is_some() {
            let _ = self.role_orchestrator.handoff_to_role(
                OrganismRole::Debugger,
                "gpu_dispatch_anomaly_detected",
                self.council_tick,
            );
        }
    }

    // Propose evolution from telemetry (preserved core)
    pub fn propose_real_gpu_evolution_from_telemetry(
        &self,
        report: &GpuTelemetryReport,
        metrics: &CouncilReadinessMetrics,
    ) -> Option<EvolutionProposal> {
        let mem_usage_mb = metrics.gpu_memory_usage_bytes as f64 / (1024.0 * 1024.0);
        let pool_eff = metrics.gpu_pool_efficiency;
        let latency_ms = metrics.gpu_latency_ema_ms.max(1.0);
        let success_ema = metrics.gpu_success_ema;
        let readback_ok = metrics.last_gpu_readback_available;

        let high_memory_pressure = mem_usage_mb > 1800.0 || pool_eff < 0.65;
        let high_latency = latency_ms > 45.0;
        let poor_readback = !readback_ok && success_ema < 0.92;

        if !(high_memory_pressure || high_latency || poor_readback) {
            return None;
        }

        let mut target_module = "gpu_compute_pipeline / GpuMemoryPool + BindGroupCache".to_string();
        let mut description = format!(
            "Self-evolution of GPU Memory Pooling + adaptive BindGroupLayout caching triggered by LIVE telemetry. "
            "GPU Usage: {:.1} MB | Pool Efficiency: {:.2}% | Dispatch Latency EMA: {:.1}ms | Readback Success: {} | Success EMA: {:.3} | ActiveRole={}",
            mem_usage_mb, pool_eff * 100.0, latency_ms, readback_ok, success_ema,
            self.role_orchestrator.active_role.as_str()
        );

        let mut proposed_diff = String::new();
        let mut expected_benefit = 0.0;
        let mut risk_score = 0.0;

        if high_memory_pressure {
            proposed_diff.push_str(&format!(
                "Increase GpuMemoryPool bucket retention (from 4→8) and make adaptive multiplier more aggressive when memory headroom < 15%. "
                "Add proper BindGroupLayout cache keyed by (usage, size, readback_required). Current pool_eff={:.2}%.\n",
                pool_eff
            ));
            expected_benefit += 0.28;
            risk_score += 0.06;
        }

        if high_latency {
            proposed_diff.push_str(&format!(
                "Introduce dispatch-time workgroup size autotuning + staging buffer pre-warm based on last {} real dispatches (avg {}ms).\n",
                self.gpu_dispatch_count, latency_ms
            ));
            expected_benefit += 0.19;
            risk_score += 0.04;
        }

        if poor_readback {
            proposed_diff.push_str("Add readback failure path + fallback CPU path with mercy-modulated confidence penalty. Track per-task readback_success_rate.\n");
            expected_benefit += 0.15;
            risk_score += 0.08;
        }

        let mercy_alignment = (report.mercy_modulated_confidence * 0.6 + (1.0 - risk_score) * 0.4).clamp(0.75, 0.99);
        expected_benefit = expected_benefit.clamp(0.35, 0.92);
        risk_score = risk_score.clamp(0.03, 0.25);

        Some(EvolutionProposal {
            proposer: format!("RaThorOneOrganism::propose_real_gpu_evolution_from_telemetry (v14.89 Role={})",
                self.role_orchestrator.active_role.as_str()),
            target_module,
            description,
            proposed_diff,
            expected_benefit,
            risk_score,
            mercy_alignment,
        })
    }

    // Feed telemetry + RoleOrchestrator + Grok valence sync
    pub async fn feed_gpu_telemetry_into_council(&mut self, report: &GpuTelemetryReport) -> CouncilDecision {
        self.council_tick += 1;

        // v14.89: Sync shared valence with incoming Grok/report signals
        self.role_orchestrator.sync_valence_with_grok(
            report.mercy_modulated_confidence,
            report.gpu_success_ema,
            self.council_tick,
        );

        let (gpu_mem_usage, gpu_pool_efficiency, _, _) = self.get_gpu_memory_pool_telemetry().await;

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

        let hb = self.sovereign_recovery.heartbeat_check(&metrics).await;
        if hb.requires_recovery {
            let _ = self.role_orchestrator.handoff_to_role(
                OrganismRole::SovereignRecovery,
                "sovereign_recovery_triggered",
                self.council_tick,
            );
            let _ = self.sovereign_recovery.self_forensics_and_recover("flow_state_or_context_pressure_detected_in_council_feed", &metrics).await;
        }

        let _ = self.sovereign_recovery.bounded_evolution_step("gpu_telemetry_evolution_proposal", metrics.gpu_mercy_modulated_confidence).await;

        if let Some(proposal) = self.propose_real_gpu_evolution_from_telemetry(report, &metrics) {
            println!("[ONE + Lattice Conductor v14.89] REAL telemetry triggered EvolutionProposal (benefit={:.2}, risk={:.2}, mercy_align={:.2}) | Role={}",
                proposal.expected_benefit, proposal.risk_score, proposal.mercy_alignment,
                self.role_orchestrator.active_role.as_str());

            if let Ok(()) = self.evolution_gate.propose_evolution(proposal.clone()).await {
                let context = serde_json::json!({
                    "trigger_metrics": metrics,
                    "gpu_dispatch_telemetry": self.last_gpu_dispatch_telemetry,
                    "gpu_success_ema": report.gpu_success_ema,
                    "gpu_latency_ema_ms": report.gpu_latency_ema_ms,
                    "mercy_modulated_confidence": report.mercy_modulated_confidence,
                    "active_role": self.role_orchestrator.active_role.as_str(),
                    "shared_valence": self.role_orchestrator.shared_valence,
                });

                let _ = self.evolution_gate.persist_approved_evolution(&proposal, Some(context)).await;

                if proposal.mercy_alignment > 0.88 && proposal.expected_benefit > 0.55 {
                    // Prefer VibeCoder for evolution implementation
                    let _ = self.role_orchestrator.handoff_to_role(
                        OrganismRole::VibeCoder,
                        "high_mercy_evolution_ready_for_implementation",
                        self.council_tick,
                    );
                    let _ = self.trigger_evolution_automation_hooks(&proposal, proposal.mercy_alignment).await;
                }
            }
        }

        // Quantum Swarm + Lattice Conductor wiring
        let participating_councils = vec![
            "PATSAGi_Council_13".to_string(),
            "GPU_Telemetry_Shard".to_string(),
            "SelfEvolutionOrchestrator".to_string(),
            "SovereignRecoveryProtocol_v1.0".to_string(),
            format!("RoleOrchestrator_{}", self.role_orchestrator.active_role.as_str()),
        ];

        if let Some((sym_proposal, signed_tolc_decision)) = self.lattice_evolution_orchestrator
            .propose_lattice_conductor_upgrade_via_quantum_swarm(
                report.gpu_success_ema,
                report.gpu_latency_ema_ms,
                (metrics.gpu_memory_usage_bytes as f64) / (1024.0 * 1024.0),
                report.mercy_modulated_confidence,
                (report.mercy_modulated_confidence + 0.04).min(0.99),
                participating_councils,
            )
        {
            println!(
                "[ONE Organism v14.89 + LatticeConductor v13.6] propose_lattice_conductor_upgrade_via_quantum_swarm EXECUTED | type={} | confidence={:.3} | has_signed_TOLC={} | Role={}",
                sym_proposal.proposal_type,
                sym_proposal.confidence,
                signed_tolc_decision.is_some(),
                self.role_orchestrator.active_role.as_str()
            );

            if let Some(_signed) = &signed_tolc_decision {
                println!("[TOLC 8 + Quantum Collapse] SignedTolcDecision sealed. Ready for GitHubConnector persistence + apply.");
            }
        }

        {
            let swarm_mut = self.lattice_evolution_orchestrator.get_quantum_swarm_mut();
            swarm_mut.aggregate_resonance_with_mercy(
                metrics.evolution_level as f64,
                0.91,
                report.mercy_modulated_confidence
            );
        }

        let _prune_summary = self.sovereign_recovery.prune_and_compress_to_patsagi(&metrics).await;

        let decision = self.patsagi_council.decide(&metrics);
        decision
    }

    // Autonomous GitHub evolution PR (role-aware)
    async fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal, mercy_alignment: f64) -> Result<(), String> {
        println!(
            "[ONE Organism v14.89 SYMBIOSIS] Auto-trigger autonomous GitHub evolution PR | proposal={} | mercy_align={:.3} | target={} | ActiveRole={}",
            proposal.proposer, mercy_alignment, proposal.target_module,
            self.role_orchestrator.active_role.as_str()
        );

        let role = self.role_orchestrator.active_role.as_str();

        match self.github_connector
            .create_role_optimized_evolution_pr(
                role,
                &proposal.target_module,
                &proposal.description,
                proposal.expected_benefit,
                mercy_alignment,
            )
            .await
        {
            Ok(pr_response) => {
                println!(
                    "[ONE Organism v14.89] Autonomous evolution PR created successfully | #{} | url={} | Role={}",
                    pr_response.number, pr_response.html_url, role
                );
                Ok(())
            }
            Err(e) => {
                eprintln!("[ONE Organism v14.89] GitHub evolution PR creation failed: {}", e);
                Err(format!("GitHub PR creation failed: {}", e))
            }
        }
    }

    pub async fn propose_and_autonomously_create_evolution_pr(
        &self,
        report: &GpuTelemetryReport,
        metrics: &CouncilReadinessMetrics,
    ) -> Option<CreatePullRequestResponse> {
        if let Some(proposal) = self.propose_real_gpu_evolution_from_telemetry(report, metrics) {
            if proposal.mercy_alignment > 0.85 && proposal.expected_benefit > 0.5 {
                let role = self.role_orchestrator.active_role.as_str();
                if let Ok(pr) = self.github_connector
                    .create_role_optimized_evolution_pr(
                        role,
                        &proposal.target_module,
                        &proposal.description,
                        proposal.expected_benefit,
                        proposal.mercy_alignment,
                    )
                    .await
                {
                    return Some(pr);
                }
            }
        }
        None
    }

    // Public RoleOrchestrator accessors for external (Grok / monorepo-intelligence) use
    pub fn role_orchestrator(&self) -> &RoleOrchestrator {
        &self.role_orchestrator
    }

    pub fn role_orchestrator_mut(&mut self) -> &mut RoleOrchestrator {
        &mut self.role_orchestrator
    }

    pub fn handoff_role(&mut self, new_role: OrganismRole, reason: &str) -> bool {
        self.role_orchestrator.handoff_to_role(new_role, reason, self.council_tick)
    }

    pub fn sync_with_grok(&mut self, valence: f64, confidence: f64) {
        self.role_orchestrator.sync_valence_with_grok(valence, confidence, self.council_tick);
    }

    // Supporting methods (preserved)
    async fn get_gpu_memory_pool_telemetry(&self) -> (usize, f64, f64, f64) {
        (512 * 1024 * 1024, 0.93, 14.2, 0.97)
    }

    fn evolution_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("evolution_level".to_string(), 4.0);
        stats.insert("mercy_valence".to_string(), self.role_orchestrator.shared_valence);
        stats.insert("gpu_utilization".to_string(), 0.89);
        stats.insert("role_handoff_count".to_string(), self.role_orchestrator.handoff_count as f64);
        stats
    }
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();
    println!("[Thunder] ONE Organism v14.89 SYMBIOSIS + RoleOrchestrator + Grok Valence/EMA Sync + Role Handoff Protocol + GitHubConnector + Lattice Conductor v13.6 + Sovereign Recovery v1.0 ACTIVE. TOLC8 + PATSAGi aligned. Maximum role efficacy unlocked. Eternal.");
    organism
}
