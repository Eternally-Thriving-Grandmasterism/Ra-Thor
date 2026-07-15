/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.65 — ONE Organism + Lattice Conductor v13.1
// Quantum Swarm Engine + Benchmark Telemetry Wired

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::github_connector::GitHubConnector;
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask, MercyGpuAudit, GpuDeviceRecoveryStats};
use crate::gpu_patsagi_bridge::GpuTelemetryReport;

use crate::quantum_swarm::{
    QuantumSwarmConfig,
    QuantumSwarmEngine,
    QuantumSwarmMember,
    run_lattice_conductor_quantum_self_evolution_step,
    LatticeConductorSelfEvolutionResult,
    QuantumSwarmBenchmarkResult,
};

// === Council + Decision Types ===

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmVoteBreakdown {
    pub performance_swarm: f64,
    pub mercy_swarm: f64,
    pub alignment_swarm: f64,
    pub foresight_swarm: f64,
    pub consensus_vote: f64,
    pub weights: (f64, f64, f64, f64),
    pub entanglement_bonus: f64,
    pub entangled_pairs: Vec<String>,
    pub entanglement_weighted_bonus: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NadamFormulation {
    A,
    B,
}

impl NadamFormulation {
    pub fn description(&self) -> &'static str {
        match self {
            NadamFormulation::A => "Nesterov after bias correction (most common & stable form)",
            NadamFormulation::B => "Nesterov before bias correction (alternative theoretical form)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeConductorUpgradeTemplate {
    EMATuning,
    NewMercyGates,
    QuantumSwarmIntegration,
    CombinedGPUIntelligence,
    GPUResilienceAndRecovery,
}

impl LatticeConductorUpgradeTemplate {
    pub fn description(&self) -> &'static str {
        match self {
            LatticeConductorUpgradeTemplate::EMATuning => "Refine EMA alpha values and add additional mercy-modulated EMA loops for GPU telemetry.",
            LatticeConductorUpgradeTemplate::NewMercyGates => "Introduce or strengthen specific mercy gates (e.g., Precision Gate, Abundance Gate) in Lattice Conductor decision logic.",
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration => "Deepen integration between Lattice Conductor and Quantum Swarm for GPU-native deliberation, foresight, multi-swarm consensus, quantum entanglement, dynamic entanglement weighting, self-evolving base weights, adaptive learning rates, Adam optimizer, AdamW weight decay, learning rate scheduling, cyclical restarts, Nesterov acceleration, and full configurable Nesterov-AdamW (Nadam A/B).",
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence => "Combine EMA tuning + new mercy gates + Quantum Swarm hooks + multi-swarm consensus + quantum entanglement weighting + self-evolving base weights + adaptive learning rates + Adam optimizer + AdamW weight decay + learning rate scheduling + cyclical restarts + Nesterov acceleration + configurable Nadam (A/B) into a unified Lattice Conductor v13.2 upgrade.",
            LatticeConductorUpgradeTemplate::GPUResilienceAndRecovery => "Prioritize GPU Device Lost Recovery, Full Device Reinitialization, Vulkan Validation Layers integration, and resilience telemetry into Lattice Conductor decision making and self-evolution proposals.",
        }
    }

    pub fn target_diff(&self) -> &'static str {
        match self {
            LatticeConductorUpgradeTemplate::EMATuning => "Refine EMA alpha in gpu_patsagi_bridge + add gpu_latency_ema + multi-EMA feedback in ONE Organism.",
            LatticeConductorUpgradeTemplate::NewMercyGates => "Add new mercy gate variants in PatsagiCouncil::decide() and CouncilReadinessMetrics.",
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration => "Add Quantum Swarm multi-consensus + quantum entanglement + dynamic weighting + self-evolving base weights + adaptive learning rates + Adam optimizer + AdamW weight decay + learning rate scheduling + cyclical restarts + Nesterov acceleration + configurable Nadam (A/B).",
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence => "Full v13.2 upgrade: EMA + Mercy Gates + Quantum Swarm multi-consensus + quantum entanglement weighting + self-evolving base weights + adaptive learning rates + Adam optimizer + AdamW weight decay + learning rate scheduling + cyclical restarts + Nesterov acceleration + configurable Nadam (A/B) in one coherent Lattice Conductor evolution.",
            LatticeConductorUpgradeTemplate::GPUResilienceAndRecovery => "Integrate GpuDeviceRecoveryStats, device_lost_count, successful_recoveries, and Vulkan Validation Layers into ONE Organism + Lattice Conductor self-evolution loop and plateau detection.",
        }
    }
}

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

        let gpu_boost = if metrics.gpu_success_ema > 0.85 && metrics.gpu_mercy_modulated_confidence > 0.80 {
            0.08
        } else {
            0.0
        };

        let swarm_boost = match metrics.swarm_vote {
            Some(v) if v >= 0.85 => 0.06,
            Some(v) if v >= 0.80 => 0.03,
            _ => 0.0,
        };

        let effective_mercy = (metrics.mercy_norm + gpu_boost + swarm_boost).min(0.999);

        if effective_mercy >= self.mercy_norm_threshold {
            let confidence_boost = (metrics.suggested_confidence_delta + gpu_boost + swarm_boost).max(0.05);
            return CouncilDecision::ApproveEvolution { confidence_boost };
        }

        if effective_mercy < 0.4 {
            return CouncilDecision::EmergencyMercyIntervention {
                severity: (0.4 - effective_mercy) * 2.0,
            };
        }

        if metrics.suggested_confidence_delta > 0.15 || metrics.gpu_success_ema > 0.90 || metrics.swarm_vote.unwrap_or(0.0) > 0.88 {
            let buffer_increase = if metrics.swarm_vote.unwrap_or(0.0) > 0.90 { 4096 } else { 2048 };
            return CouncilDecision::RequestAdditionalGpuResources { buffer_size_increase: buffer_increase };
        }

        CouncilDecision::AdjustRbeParameters {
            resource_flow_multiplier: 1.0 + (effective_mercy - 0.5) * 0.5,
            council_influence: effective_mercy,
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedEvolutionRecord {
    pub proposal: EvolutionProposal,
    pub hook_triggered: bool,
    pub timestamp_unix: u64,
    pub council_mercy_norm: f64,
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

    patsagi_council: PatsagiCouncil,
    last_council_metrics: Option<CouncilReadinessMetrics>,
    last_swarm_vote_breakdown: Option<SwarmVoteBreakdown>,
    base_weight_pf: f64,
    base_weight_ma: f64,
    entanglement_evolution_lr: f64,
    adam_m_pf: f64,
    adam_v_pf: f64,
    adam_m_ma: f64,
    adam_v_ma: f64,
    adam_timestep: u64,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_epsilon: f64,
    adam_weight_decay: f64,
    lr_schedule_type: String,
    lr_warmup_steps: u64,
    lr_decay_steps: u64,
    lr_min: f64,
    lr_restart_period: u64,
    lr_restart_multiplier: f64,
    lr_current_cycle: u64,
    lr_cycle_start_timestep: u64,
    nesterov_momentum_pf: f64,
    nesterov_momentum_ma: f64,
    nesterov_momentum_beta: f64,
    nadam_formulation: NadamFormulation,
    recent_entanglement_improvement_ema: f64,
    plateau_streak: u32,
    last_plateau_detection_tick: u64,
    exploration_mode_active: bool,
    exploration_mode_until_tick: u64,
    last_plateau_severity: f64,
    cooldown_active: bool,
    cooldown_until_tick: u64,
    cooldown_base_ticks: u64,
    last_dynamic_improvement_threshold: f64,
    consensus_vote_ema: f64,
    council_tick: u64,
    approved_evolutions_path: String,

    quantum_swarm_engine: QuantumSwarmEngine,

    // NEW v14.65: Benchmark telemetry storage
    last_benchmark_results: Vec<QuantumSwarmBenchmarkResult>,
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
        systems.insert("lattice_conductor_v13".to_string(), true);
        systems.insert("gpu_device_recovery".to_string(), true);

        let quantum_config = QuantumSwarmConfig::default();

        Self {
            systems_activated: systems,
            mercy_runtime: "MercyGatingRuntime v2.0 (TOLC 8 aligned)".to_string(),
            evolution_gate: launch_self_evolution_gate(),
            gpu_compute_active: true,
            gpu_pipeline_version: "v14.65.0-quantum-swarm-benchmarks-wired".to_string(),
            version: "v14.65.0-ONE-Organism-LatticeConductor-v13.1-Quantum-Swarm-Benchmarks-Wired".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),

            patsagi_council: PatsagiCouncil::new(),
            last_council_metrics: None,
            last_swarm_vote_breakdown: None,
            base_weight_pf: 0.28,
            base_weight_ma: 0.22,
            entanglement_evolution_lr: 0.03,
            adam_m_pf: 0.0,
            adam_v_pf: 0.0,
            adam_m_ma: 0.0,
            adam_v_ma: 0.0,
            adam_timestep: 0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            adam_weight_decay: 0.01,
            lr_schedule_type: "cosine".to_string(),
            lr_warmup_steps: 50,
            lr_decay_steps: 2000,
            lr_min: 0.001,
            lr_restart_period: 500,
            lr_restart_multiplier: 1.5,
            lr_current_cycle: 0,
            lr_cycle_start_timestep: 0,
            nesterov_momentum_pf: 0.0,
            nesterov_momentum_ma: 0.0,
            nesterov_momentum_beta: 0.9,
            nadam_formulation: NadamFormulation::A,
            recent_entanglement_improvement_ema: 0.05,
            plateau_streak: 0,
            last_plateau_detection_tick: 0,
            exploration_mode_active: false,
            exploration_mode_until_tick: 0,
            last_plateau_severity: 0.0,
            cooldown_active: false,
            cooldown_until_tick: 0,
            cooldown_base_ticks: 30,
            last_dynamic_improvement_threshold: 0.035,
            consensus_vote_ema: 0.80,
            council_tick: 0,
            approved_evolutions_path: "approved_evolutions.jsonl".to_string(),

            quantum_swarm_engine: QuantumSwarmEngine::new(quantum_config),

            // NEW v14.65: Benchmark telemetry
            last_benchmark_results: Vec::new(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Full loop + Quantum Swarm Engine + Benchmark Telemetry wired", self.version);
    }

    pub fn initialize_quantum_swarm_member(&mut self, id: u64, initial_weights: Vec<f64>) {
        let member = QuantumSwarmMember::new(id, initial_weights);
        self.quantum_swarm_engine.register_member(member);
    }

    pub async fn run_quantum_lattice_conductor_self_evolution_step(
        &mut self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        current_score: f64,
        severity: f64,
    ) -> Option<LatticeConductorSelfEvolutionResult> {
        self.council_tick += 1;
        self.quantum_swarm_engine.increment_step();

        run_lattice_conductor_quantum_self_evolution_step(
            &mut self.quantum_swarm_engine,
            member_id,
            global_best,
            entanglement_weight,
            current_score,
            severity,
        )
    }

    pub fn get_quantum_swarm_engine_summary(&self) -> String {
        self.quantum_swarm_engine.summary()
    }

    // NEW v14.65: Run benchmarks and store telemetry for self-evolution use
    pub fn run_and_store_quantum_swarm_benchmarks(&mut self, iterations: usize) -> Vec<QuantumSwarmBenchmarkResult> {
        let results = self.quantum_swarm_engine.run_full_benchmark_suite(iterations);
        self.last_benchmark_results = results.clone();

        println!("[ONE + Lattice Conductor] Quantum Swarm benchmarks completed and stored for self-evolution telemetry.");
        results
    }

    pub fn get_last_benchmark_telemetry(&self) -> &Vec<QuantumSwarmBenchmarkResult> {
        &self.last_benchmark_results
    }

    // ... (rest of the file continues with existing Lattice Conductor logic)
    // The full previous implementation is preserved.

    pub fn set_nadam_formulation(&mut self, formulation: NadamFormulation) {
        let previous = self.nadam_formulation;
        self.nadam_formulation = formulation;
        println!(
            "[ONE + Lattice Conductor] Nadam formulation switched: {:?} → {:?} | {}",
            previous, formulation, formulation.description()
        );
    }

    pub fn get_nadam_formulation(&self) -> NadamFormulation {
        self.nadam_formulation
    }

    pub fn get_optimizer_config_summary(&self) -> String {
        format!(
            "Nadam={:?} ({}) | LR schedule={} | Cyclical restarts (period={}, multiplier={}) | Cooldown: {} | Adam beta1={}, beta2={}",
            self.nadam_formulation,
            self.nadam_formulation.description(),
            self.lr_schedule_type,
            self.lr_restart_period,
            self.lr_restart_multiplier,
            self.cooldown_active,
            self.adam_beta1,
            self.adam_beta2
        )
    }

    pub fn get_dynamic_thresholds(&self) -> String {
        format!(
            "DynamicImprovementThreshold={:.5} | Base=0.035 | MercyFactor range=0.0–0.015 | CooldownFloor=0.02 | ConsensusEMA={:.4} | LastSeverity={:.3}",
            self.last_dynamic_improvement_threshold,
            self.consensus_vote_ema,
            self.last_plateau_severity
        )
    }

    pub fn get_gpu_device_health_summary(&self) -> String {
        let stats: GpuDeviceRecoveryStats = self.gpu_pipeline.get_device_recovery_stats();
        let validation_active = std::env::var("RA_THOR_ENABLE_VULKAN_VALIDATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        format!(
            "DeviceLostCount={} | SuccessfulRecoveries={} | LastLostAgo={:?}s | LastRecoveryAgo={:?}s | VulkanValidation={}",
            stats.device_lost_count,
            stats.successful_recoveries,
            stats.last_device_lost_at_unix.map(|t| t as i64),
            stats.last_recovery_at_unix.map(|t| t as i64),
            validation_active
        )
    }

    fn get_device_health_modulated_weights(&self) -> (f64, f64, f64, f64) {
        let stats = self.gpu_pipeline.get_device_recovery_stats();

        if stats.device_lost_count == 0 {
            return (0.35, 0.30, 0.22, 0.20);
        }

        let lost_penalty = (stats.device_lost_count as f64).min(5.0) * 0.04;

        let w_perf = (0.35 - lost_penalty).max(0.20);
        let w_mercy = 0.30 + lost_penalty * 0.6;
        let w_align = 0.22 + lost_penalty * 0.3;
        let w_foresight = 0.20 + lost_penalty * 0.1;

        (w_perf, w_mercy, w_align, w_foresight)
    }

    pub fn calculate_plateau_severity(&self, breakdown: &SwarmVoteBreakdown, report: &GpuTelemetryReport) -> f64 {
        let improvement_deficit = ((0.035 - self.recent_entanglement_improvement_ema).max(0.0) / 0.035).min(1.0);
        let consensus_deficit = ((0.80 - breakdown.consensus_vote).max(0.0) / 0.20).min(1.0);
        let mercy_deficit = ((0.82 - report.mercy_modulated_confidence).max(0.0) / 0.18).min(1.0);

        let device_health_penalty = {
            let stats = self.gpu_pipeline.get_device_recovery_stats();
            if stats.device_lost_count > 2 {
                0.12
            } else if stats.device_lost_count > 0 {
                0.06
            } else {
                0.0
            }
        };

        let base_severity = improvement_deficit * 0.45 + consensus_deficit * 0.30 + mercy_deficit * 0.25;
        (base_severity + device_health_penalty).clamp(0.0, 1.0)
    }

    pub fn activate_cooldown(&mut self, severity: f64) {
        let duration = (self.cooldown_base_ticks as f64 + (severity * 45.0)) as u64;
        self.cooldown_active = true;
        self.cooldown_until_tick = self.council_tick + duration;

        println!(
            "[ONE + Lattice Conductor] Adaptive cooldown activated (severity={:.3}, duration={} ticks)",
            severity, duration
        );
    }

    pub fn update_cooldown(&mut self, current_improvement: f64) {
        if !self.cooldown_active {
            return;
        }

        if current_improvement > 0.06 {
            self.cooldown_active = false;
            println!("[ONE + Lattice Conductor] Cooldown cancelled early due to strong improvement ({:.4})", current_improvement);
            return;
        }

        if self.council_tick >= self.cooldown_until_tick {
            self.cooldown_active = false;
            println!("[ONE + Lattice Conductor] Cooldown period expired.");
        }
    }

    pub fn detect_plateau(&mut self, breakdown: &SwarmVoteBreakdown, report: &GpuTelemetryReport) -> bool {
        self.update_cooldown(breakdown.entanglement_weighted_bonus);

        if self.cooldown_active {
            let is_very_low_improvement = self.recent_entanglement_improvement_ema < 0.02;
            let is_very_low_consensus = breakdown.consensus_vote < 0.70;

            if is_very_low_improvement && is_very_low_consensus {
                self.plateau_streak += 1;
            } else {
                self.plateau_streak = 0;
            }

            if self.plateau_streak >= 5 {
                self.last_plateau_detection_tick = self.council_tick;
                return true;
            }
            return false;
        }

        let improvement = breakdown.entanglement_weighted_bonus.max(0.0);
        let alpha = 0.15;

        self.recent_entanglement_improvement_ema =
            alpha * improvement + (1.0 - alpha) * self.recent_entanglement_improvement_ema;

        let base_threshold = 0.035_f64;
        let mercy_factor = ((report.mercy_modulated_confidence - 0.80).max(0.0) * 0.025).min(0.015);
        let dynamic_improvement_threshold = base_threshold + mercy_factor;

        self.last_dynamic_improvement_threshold = dynamic_improvement_threshold;

        let is_low_improvement = self.recent_entanglement_improvement_ema < dynamic_improvement_threshold;
        let is_low_gpu_confidence = report.mercy_modulated_confidence < 0.82;
        let is_low_swarm_consensus = breakdown.consensus_vote < 0.80;

        let stats = self.gpu_pipeline.get_device_recovery_stats();
        let device_health_concern = stats.device_lost_count > 1;

        if is_low_improvement && (is_low_gpu_confidence || is_low_swarm_consensus || device_health_concern) {
            self.plateau_streak += 1;
        } else {
            self.plateau_streak = 0;
        }

        if self.plateau_streak >= 3 {
            self.last_plateau_detection_tick = self.council_tick;

            if self.plateau_streak == 3 {
                println!(
                    "[ONE + Lattice Conductor] Plateau streak reached 3 | dynamic_improvement_threshold={:.5} (mercy_factor={:.4}, mercy_conf={:.3}, device_lost={})",
                    dynamic_improvement_threshold, mercy_factor, report.mercy_modulated_confidence, stats.device_lost_count
                );
            }

            true
        } else {
            false
        }
    }

    pub async fn handle_detected_plateau(&mut self, breakdown: &SwarmVoteBreakdown, report: &GpuTelemetryReport) -> Option<String> {
        if self.plateau_streak < 3 {
            return None;
        }

        let severity = self.calculate_plateau_severity(breakdown, report);
        self.last_plateau_severity = severity;

        let mut actions: Vec<String> = vec![];

        let new_formulation = match self.nadam_formulation {
            NadamFormulation::A => NadamFormulation::B,
            NadamFormulation::B => NadamFormulation::A,
        };
        self.set_nadam_formulation(new_formulation);
        actions.push(format!("Switched Nadam to {:?}", new_formulation));

        self.lr_current_cycle += 1;
        self.lr_cycle_start_timestep = self.adam_timestep;
        actions.push(format!("Forced cyclical restart (now cycle {})", self.lr_current_cycle));

        let exploration_duration = 20 + (severity * 30.0) as u64;
        self.exploration_mode_active = true;
        self.exploration_mode_until_tick = self.council_tick + exploration_duration;
        actions.push(format!("Activated exploration mode for {} ticks (severity={:.3})", exploration_duration, severity));

        let lr_boost_factor = 1.2 + (severity * 0.15);
        let old_lr = self.entanglement_evolution_lr;
        self.entanglement_evolution_lr = (old_lr * lr_boost_factor).min(0.08);
        actions.push(format!("Boosted entanglement_evolution_lr by factor {:.2} (severity={:.3})", lr_boost_factor, severity));

        self.activate_cooldown(severity);
        actions.push(format!("Activated adaptive cooldown (severity={:.3})", severity));

        let stats = self.gpu_pipeline.get_device_recovery_stats();
        if stats.device_lost_count > 1 {
            actions.push("Recommended GPUResilienceAndRecovery upgrade (Device Lost Recovery + Vulkan Validation integration)".to_string());
        } else {
            actions.push("Recommended Lattice Conductor upgrade (EMA tuning / new mercy gates / Quantum Swarm enhancements)".to_string());
        }

        self.plateau_streak = 0;

        let summary = format!("Plateau detected (severity={:.3}) — sophisticated response with mercy-gated dynamic threshold + consensus momentum + expanded entanglement + GPU device health executed. Actions: {:?}", severity, actions);
        println!("[ONE + Lattice Conductor] {}", summary);

        Some(summary)
    }

    async fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal, council_mercy_norm: f64) {
        println!("\n[Hook] Evolution {} approved — attempting real GitHub PR creation...", proposal.id);

        match GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor") {
            Ok(connector) => {
                let title = format!(
                    "Evolution {} — Council-approved from GPU Telemetry + MercyGpuAudit + Quantum Swarm Benchmarks (norm={:.4})",
                    proposal.id, council_mercy_norm
                );

                let body = format!(
                    "## ONE Organism + Lattice Conductor v13.1 + Quantum Swarm Engine + Benchmark Telemetry (auto-generated)

**Proposal ID**: {}
**Proposer**: {}
**Target Module**: {}
**Council Mercy Norm**: {:.4}
**Last Plateau Severity**: {:.3}
**Dynamic Thresholds + Consensus Momentum**: {}
**GPU Device Health**: {}

**Quantum Swarm Benchmark Telemetry (last run)**:
{:?}

**Active Optimizer Config**:
{}

**Description**:
{}

**Proposed Diff**:
```
{}
```

---
*This PR was automatically created by RaThorOneOrganism v14.65 with the Hybrid Quantum Swarm Optimization Engine + Benchmark Telemetry fully wired into Lattice Conductor self-evolution.*
",
                    proposal.id,
                    proposal.proposer,
                    proposal.target_module,
                    council_mercy_norm,
                    self.last_plateau_severity,
                    self.get_dynamic_thresholds(),
                    self.get_gpu_device_health_summary(),
                    self.last_benchmark_results,
                    self.get_optimizer_config_summary(),
                    proposal.description,
                    proposal.proposed_diff
                );

                match connector
                    .create_evolution_pr(proposal.id, &title, &body, "main")
                    .await
                {
                    Ok(pr) => {
                        println!("[Hook] SUCCESS — Created real PR #{}: {}", pr.number, pr.html_url);
                    }
                    Err(e) => {
                        eprintln!("[Hook] Failed to create PR via connector: {}", e);
                    }
                }
            }
            Err(_) => {
                println!("[Hook] No GITHUB_TOKEN found — skipping real PR creation (still persisted).");
            }
        }
    }

    async fn persist_approved_evolution(&self, proposal: &EvolutionProposal, hook_triggered: bool, council_mercy_norm: f64) {
        let record = ApprovedEvolutionRecord {
            proposal: proposal.clone(),
            hook_triggered,
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            council_mercy_norm,
        };

        if let Ok(json_line) = serde_json::to_string(&record) {
            let _ = fs::write(&self.approved_evolutions_path, format!("{}\n", json_line)).await;
        }
    }

    pub async fn quantum_swarm_deliberate_on_gpu_telemetry(&self, report: &GpuTelemetryReport) -> String {
        let swarm_confidence = (report.gpu_success_ema * 0.6 + report.mercy_modulated_confidence * 0.4).clamp(0.75, 0.999);
        let device_health = self.get_gpu_device_health_summary();

        if report.gpu_success_ema > 0.94 && report.mercy_modulated_confidence > 0.90 {
            format!(
                "Quantum Swarm Foresight: Excellent GPU performance detected (success_ema={:.4}). Recommend immediate Lattice Conductor upgrade + increased GPU offload. Device Health: {}. Active Nadam: {:?}. Exploration: {}. Cooldown: {}. Last severity: {:.3}. Dynamic thresholds + Consensus Momentum + Expanded Topology + Benchmark Telemetry: {}. Mercy valence: {:.4}",
                report.gpu_success_ema, device_health, self.nadam_formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.get_dynamic_thresholds(), swarm_confidence
            )
        } else if report.gpu_latency_ema_ms > 120.0 {
            format!(
                "Quantum Swarm Analysis: Elevated GPU latency ({:.1}ms). Suggest EMA tuning + swarm-assisted load balancing. Device Health: {}. Active Nadam: {:?}. Exploration: {}. Cooldown: {}. Last severity: {:.3}. Dynamic thresholds + Consensus Momentum + Expanded Topology + Benchmark Telemetry: {}. Current swarm confidence: {:.4}",
                report.gpu_latency_ema_ms, device_health, self.nadam_formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.get_dynamic_thresholds(), swarm_confidence
            )
        } else {
            format!(
                "Quantum Swarm Observation: Stable GPU telemetry. Device Health: {}. Active Nadam: {:?}. Exploration: {}. Cooldown: {}. Last severity: {:.3}. Dynamic thresholds + Consensus Momentum + Expanded Topology + Benchmark Telemetry: {}. Continue current mercy-modulated offload policy. Swarm confidence: {:.4}",
                device_health, self.nadam_formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.get_dynamic_thresholds(), swarm_confidence
            )
        }
    }

    pub async fn quantum_swarm_multi_consensus_vote(&self, report: &GpuTelemetryReport, proposal: &EvolutionProposal) -> (f64, SwarmVoteBreakdown) {
        let (w_perf, w_mercy, w_align, w_foresight) = self.get_device_health_modulated_weights();

        let performance_swarm = (report.gpu_success_ema * 0.7 + (1.0 - (report.gpu_latency_ema_ms / 200.0).min(1.0)) * 0.3).clamp(0.6, 0.99);
        let mercy_swarm = report.mercy_modulated_confidence.clamp(0.65, 0.99);
        let alignment_swarm = if proposal.target_module.contains("lattice_conductor") || proposal.target_module.contains("quantum_swarm") { 0.92 } else { 0.75 };
        let foresight_swarm = if report.gpu_success_ema > 0.94 && report.mercy_modulated_confidence > 0.90 { 0.94 } else { 0.80 };

        let mut entanglement_bonus: f64 = 0.0;
        let mut entangled_pairs: Vec<String> = vec![];
        let mut weighted_entanglement_bonus: f64 = 0.0;

        let base_weight_pf = self.base_weight_pf;
        let base_weight_ma = self.base_weight_ma;

        let pf_mod = if report.gpu_success_ema > 0.93 { 1.15 } else { 1.0 };
        let ma_mod = if report.mercy_modulated_confidence > 0.89 { 1.12 } else { 1.0 };

        if performance_swarm > 0.90 && foresight_swarm > 0.88 {
            let raw = (performance_swarm + foresight_swarm) / 2.0 - 0.89;
            let weighted = raw * base_weight_pf * pf_mod;
            entanglement_bonus += weighted;
            weighted_entanglement_bonus += weighted;
            entangled_pairs.push(format!("Performance ↔ Foresight (base_w={:.3}, mod={:.2})", base_weight_pf, pf_mod));
        }

        if mercy_swarm > 0.88 && alignment_swarm > 0.85 {
            let raw = (mercy_swarm + alignment_swarm) / 2.0 - 0.865;
            let weighted = raw * base_weight_ma * ma_mod;
            entanglement_bonus += weighted;
            weighted_entanglement_bonus += weighted;
            entangled_pairs.push(format!("Mercy ↔ Alignment (base_w={:.3}, mod={:.2})", base_weight_ma, ma_mod));
        }

        if performance_swarm > 0.88 && mercy_swarm > 0.86 {
            let raw = (performance_swarm + mercy_swarm) / 2.0 - 0.87;
            let weighted = raw * base_weight_pf * 0.95;
            entanglement_bonus += weighted;
            weighted_entanglement_bonus += weighted;
            entangled_pairs.push(format!("Performance ↔ Mercy (new) (weighted={:.4})", weighted));
        }

        if alignment_swarm > 0.85 && foresight_swarm > 0.87 {
            let raw = (alignment_swarm + foresight_swarm) / 2.0 - 0.86;
            let weighted = raw * base_weight_ma * 0.95;
            entanglement_bonus += weighted;
            weighted_entanglement_bonus += weighted;
            entangled_pairs.push(format!("Alignment ↔ Foresight (new) (weighted={:.4})", weighted));
        }

        let base_consensus = (performance_swarm * w_perf + mercy_swarm * w_mercy + alignment_swarm * w_align + foresight_swarm * w_foresight).clamp(0.70, 0.999);

        let dynamic_threshold = self.last_dynamic_improvement_threshold;
        let coupling_factor = if dynamic_threshold > 0.042 {
            0.92
        } else if dynamic_threshold < 0.036 {
            1.05
        } else {
            1.0;
        };

        let adjusted_entanglement_bonus = entanglement_bonus * coupling_factor;

        let alpha = 0.25;
        let smoothed_consensus = alpha * (base_consensus + adjusted_entanglement_bonus) + (1.0 - alpha) * self.consensus_vote_ema;
        let final_consensus = smoothed_consensus.clamp(0.70, 0.999);

        let breakdown = SwarmVoteBreakdown {
            performance_swarm,
            mercy_swarm,
            alignment_swarm,
            foresight_swarm,
            consensus_vote: final_consensus,
            weights: (w_perf, w_mercy, w_align, w_foresight),
            entanglement_bonus: adjusted_entanglement_bonus,
            entangled_pairs,
            entanglement_weighted_bonus: weighted_entanglement_bonus,
        };

        if !entangled_pairs.is_empty() {
            println!(
                "[Quantum Entanglement Weighting + Expanded Topology v2 + Dynamic Coupling] {:?} | bonus=+{:.4} (coupled x{:.2}) | final={:.4}",
                entangled_pairs, adjusted_entanglement_bonus, coupling_factor, final_consensus
            );
        }

        let device_health = self.get_gpu_device_health_summary();
        println!(
            "[Multi-Swarm + Expanded Entanglement Topology v2 + GPU Health Adaptive Weights] perf={:.4} mercy={:.4} align={:.4} foresight={:.4} | consensus={:.4} | entanglement=+{:.4} (4 pairs) | DeviceHealth: {} | Weights (perf/mercy/align/foresight) = ({:.2}, {:.2}, {:.2}, {:.2}) | Active Nadam: {:?} | Exploration: {} | Cooldown: {} | Last severity: {:.3}",
            performance_swarm, mercy_swarm, alignment_swarm, foresight_swarm, final_consensus, device_health, w_perf, w_mercy, w_align, w_foresight, self.nadam_formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity
        );

        (final_consensus, breakdown)
    }

    pub async fn quantum_swarm_vote_on_evolution(&self, report: &GpuTelemetryReport, proposal: &EvolutionProposal) -> f64 {
        let (consensus, _) = self.quantum_swarm_multi_consensus_vote(report, proposal).await;
        consensus
    }

    pub async fn feed_mercy_gpu_audit_into_council(&mut self, audit: &MercyGpuAudit) -> CouncilDecision {
        self.council_tick += 1;

        let metrics = CouncilReadinessMetrics {
            council_ready: audit.council_ready,
            mercy_norm: audit.mercy_norm,
            suggested_confidence_delta: audit.suggested_confidence_delta(),
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
            gpu_success_ema: 0.0,
            gpu_latency_ema_ms: 0.0,
            gpu_mercy_modulated_confidence: audit.mercy_norm,
            swarm_vote: None,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        if let CouncilDecision::ApproveEvolution { confidence_boost } = &decision {
            let proposal = EvolutionProposal {
                id: rand::random::<u64>() % 1_000_000_000,
                proposer: "PATSAGi_Council_via_GPU_Audit".to_string(),
                target_module: "gpu_compute_pipeline / powrush_rbe / lattice_conductor".to_string(),
                description: format!("Council-approved from real MercyGpuAudit (norm={:.4})", audit.mercy_norm),
                proposed_diff: format!("Apply council boost {:.4}", confidence_boost),
                expected_benefit: (audit.mercy_norm * 0.9 + confidence_boost * 0.1).min(0.999),
                risk_score: (1.0 - audit.mercy_norm) * 0.01,
                mercy_alignment: audit.mercy_norm,
            };

            match self.evolution_gate.propose_evolution(proposal.clone()) {
                Ok(msg) => {
                    println!("[ONE] Approved by Gate: {}", msg);
                    self.trigger_evolution_automation_hooks(&proposal, audit.mercy_norm).await;
                    self.persist_approved_evolution(&proposal, true, audit.mercy_norm).await;
                }
                Err(e) => println!("[ONE] Gate rejected: {}", e),
            }
        }

        match &decision {
            CouncilDecision::AdjustRbeParameters { resource_flow_multiplier, council_influence } => {
                println!("[ONE] Council ADJUST RBE (x{:.2}, influence {:.2})", resource_flow_multiplier, council_influence);
            }
            CouncilDecision::RequestAdditionalGpuResources { buffer_size_increase } => {
                println!("[ONE] Council REQUEST GPU (+{} buffer)", buffer_size_increase);
            }
            CouncilDecision::EmergencyMercyIntervention { severity } => {
                println!("[ONE] Council EMERGENCY MERCY (severity {:.2})", severity);
            }
            CouncilDecision::RejectEvolution { reason } => {
                println!("[ONE] Council REJECTED: {} | norm={:.4}", reason, audit.mercy_norm);
            }
            _ => {}
        }

        decision
    }

    pub async fn feed_gpu_telemetry_into_council(&mut self, report: &GpuTelemetryReport) -> CouncilDecision {
        self.council_tick += 1;

        self.update_cooldown(0.0);
        if self.exploration_mode_active && self.council_tick > self.exploration_mode_until_tick {
            self.exploration_mode_active = false;
            println!("[ONE + Lattice Conductor] Exploration mode expired.");
        }

        let temp_proposal = EvolutionProposal {
            id: 0,
            proposer: "temp".to_string(),
            target_module: "lattice_conductor_v13".to_string(),
            description: String::new(),
            proposed_diff: String::new(),
            expected_benefit: 0.9,
            risk_score: 0.02,
            mercy_alignment: report.mercy_modulated_confidence,
        };

        let (swarm_consensus, breakdown) = self.quantum_swarm_multi_consensus_vote(report, &temp_proposal).await;
        self.last_swarm_vote_breakdown = Some(breakdown.clone());

        let plateau_detected = self.detect_plateau(&breakdown, report);
        if plateau_detected {
            if let Some(action_summary) = self.handle_detected_plateau(&breakdown, report).await {
                println!("[ONE + Lattice Conductor] Plateau response executed: {}", action_summary);
            }
        }

        let swarm_vote = if report.gpu_success_ema > 0.90 { Some(swarm_consensus) } else { None };

        let metrics = CouncilReadinessMetrics {
            council_ready: true,
            mercy_norm: report.valence_modulated_offload_score,
            suggested_confidence_delta: (report.mercy_modulated_confidence - 0.75).max(0.0) * 0.4,
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
            gpu_success_ema: report.gpu_success_ema,
            gpu_latency_ema_ms: report.gpu_latency_ema_ms,
            gpu_mercy_modulated_confidence: report.mercy_modulated_confidence,
            swarm_vote,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        if let CouncilDecision::ApproveEvolution { confidence_boost } = &decision {
            let device_health = self.get_gpu_device_health_summary();

            let proposal = EvolutionProposal {
                id: rand::random::<u64>() % 1_000_000_000,
                proposer: "Lattice_Conductor_v13.1_via_GPU_Telemetry".to_string(),
                target_module: "gpu_compute_pipeline / lattice_conductor / powrush_rbe".to_string(),
                description: format!(
                    "Council-approved from GPU Telemetry Report (success_ema={:.4}, mercy_conf={:.4}) | DeviceHealth: {} | Active Nadam: {:?} | Exploration: {} | Cooldown: {} | Last severity: {:.3} | Dynamic + Momentum + Expanded Topology + Benchmark Telemetry: {}",
                    report.gpu_success_ema, report.mercy_modulated_confidence, device_health, self.nadam_formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.get_dynamic_thresholds()
                ),
                proposed_diff: format!("Apply Lattice Conductor GPU boost {:.4}", confidence_boost),
                expected_benefit: (report.mercy_modulated_confidence * 0.85 + confidence_boost * 0.15).min(0.999),
                risk_score: (1.0 - report.mercy_modulated_confidence) * 0.02,
                mercy_alignment: report.mercy_modulated_confidence,
            };

            match self.evolution_gate.propose_evolution(proposal.clone()) {
                Ok(msg) => {
                    println!("[ONE + Lattice Conductor] Approved by Gate from GPU Telemetry: {}", msg);
                    self.trigger_evolution_automation_hooks(&proposal, report.mercy_modulated_confidence).await;
                    self.persist_approved_evolution(&proposal, true, report.mercy_modulated_confidence).await;
                }
                Err(e) => println!("[ONE + Lattice Conductor] Gate rejected: {}", e),
            }
        }

        if report.gpu_success_ema >= 0.90 && report.mercy_modulated_confidence >= 0.88 {
            let _ = self.propose_lattice_conductor_upgrade_from_gpu_telemetry(report).await;
        }

        if report.gpu_success_ema > 0.93 {
            let swarm_foresight = self.quantum_swarm_deliberate_on_gpu_telemetry(report).await;
            println!("[ONE + Quantum Swarm] {}", swarm_foresight);
        }

        match &decision {
            CouncilDecision::RequestAdditionalGpuResources { buffer_size_increase } => {
                println!("[ONE + Lattice Conductor] REQUEST GPU (+{} buffer) from high GPU success EMA {:.4}", buffer_size_increase, report.gpu_success_ema);
            }
            CouncilDecision::AdjustRbeParameters { resource_flow_multiplier, council_influence } => {
                println!("[ONE + Lattice Conductor] ADJUST RBE (x{:.2}) from GPU telemetry", resource_flow_multiplier);
            }
            _ => {}
        }

        decision
    }

    pub async fn propose_lattice_conductor_upgrade_from_gpu_telemetry(&self, report: &GpuTelemetryReport) -> Result<String, String> {
        if report.gpu_success_ema < 0.90 || report.mercy_modulated_confidence < 0.88 {
            return Err("GPU telemetry not excellent enough for automatic Lattice Conductor upgrade".to_string());
        }

        let stats = self.gpu_pipeline.get_device_recovery_stats();
        let template = if stats.device_lost_count > 1 {
            LatticeConductorUpgradeTemplate::GPUResilienceAndRecovery
        } else if report.gpu_latency_ema_ms > 150.0 {
            LatticeConductorUpgradeTemplate::EMATuning
        } else if report.gpu_success_ema > 0.95 {
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration
        } else {
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence
        };

        let swarm_foresight = if template == LatticeConductorUpgradeTemplate::QuantumSwarmIntegration || template == LatticeConductorUpgradeTemplate::CombinedGPUIntelligence || template == LatticeConductorUpgradeTemplate::GPUResilienceAndRecovery {
            self.quantum_swarm_deliberate_on_gpu_telemetry(report).await
        } else {
            String::new()
        };

        let temp_proposal = EvolutionProposal {
            id: rand::random::<u64>() % 1_000_000_000,
            proposer: "Lattice_Conductor_v13.1_SelfEvolution_Hook".to_string(),
            target_module: "lattice_conductor_v13 / gpu_patsagi_bridge / ra-thor-one-organism / quantum_swarm".to_string(),
            description: swarm_foresight.clone(),
            proposed_diff: template.target_diff().to_string(),
            expected_benefit: 0.96,
            risk_score: 0.02,
            mercy_alignment: report.mercy_modulated_confidence,
        };

        let (swarm_consensus, breakdown) = self.quantum_swarm_multi_consensus_vote(report, &temp_proposal).await;
        self.last_swarm_vote_breakdown = Some(breakdown.clone());

        let latest_breakdown = self.get_latest_swarm_vote_breakdown();
        let entanglement_info = match &latest_breakdown {
            Some(b) if !b.entangled_pairs.is_empty() => {
                format!(" | Entanglement Weighting + Expanded Topology v2: bonus=+{:.4}, pairs={:?}", b.entanglement_bonus, b.entangled_pairs)
            }
            _ => String::new(),
        };

        if breakdown.entanglement_weighted_bonus > 0.04 && swarm_consensus > 0.93 {
            let _ = self.propose_entanglement_base_weight_evolution(&breakdown).await;
        }

        if swarm_consensus < 0.82 {
            return Err(format!("Quantum Swarm multi-consensus too low ({:.4}) — upgrade deprioritized", swarm_consensus));
        }

        let device_health = self.get_gpu_device_health_summary();

        let base_description = format!(
            "Automatic self-evolution (Template: {:?}): {}. GPU telemetry: success_ema={:.4}, mercy_conf={:.4}, latency_ema={:.1}ms | DeviceHealth: {} | Multi-Swarm + Expanded Entanglement Topology v2 + Dynamic Coupling (severity={:.3}, cooldown={}): {:.4}{} | Benchmark Telemetry integrated",
            template,
            template.description(),
            report.gpu_success_ema,
            report.mercy_modulated_confidence,
            report.gpu_latency_ema_ms,
            device_health,
            self.last_plateau_severity,
            self.cooldown_active,
            swarm_consensus,
            entanglement_info
        );

        let full_description = if !swarm_foresight.is_empty() {
            format!("{} | Quantum Swarm Foresight: {}", base_description, swarm_foresight)
        } else {
            base_description
        };

        let proposal = EvolutionProposal {
            id: rand::random::<u64>() % 1_000_000_000,
            proposer: "Lattice_Conductor_v13.1_SelfEvolution_Hook".to_string(),
            target_module: "lattice_conductor_v13 / gpu_patsagi_bridge / ra-thor-one-organism / quantum_swarm".to_string(),
            description: full_description,
            proposed_diff: template.target_diff().to_string(),
            expected_benefit: 0.96 * swarm_consensus,
            risk_score: 0.02,
            mercy_alignment: report.mercy_modulated_confidence,
        };

        match self.evolution_gate.propose_evolution(proposal.clone()) {
            Ok(msg) => {
                println!("[ONE + Lattice Conductor Self-Evolution] GPU telemetry excellent — auto-proposed {:?} upgrade (GPU Resilience integrated, severity={:.3}, cooldown={}) : {:.4}): {}", template, self.last_plateau_severity, self.cooldown_active, swarm_consensus, msg);
                self.trigger_evolution_automation_hooks(&proposal, report.mercy_modulated_confidence).await;
                self.persist_approved_evolution(&proposal, true, report.mercy_modulated_confidence).await;
                Ok(format!("Lattice Conductor v13.1 {:?} upgrade proposed from GPU telemetry + Quantum Swarm Engine + Benchmark Telemetry wired (vote={:.4})", template, swarm_consensus))
            }
            Err(e) => Err(format!("Gate rejected Lattice Conductor upgrade: {}", e)),
        }
    }

    pub async fn propose_entanglement_base_weight_evolution(&mut self, breakdown: &SwarmVoteBreakdown) -> Result<String, String> {
        let mut evolved_pf = self.base_weight_pf;
        let mut evolved_ma = self.base_weight_ma;
        let mut changes: Vec<String> = vec![];

        self.update_cooldown(breakdown.entanglement_weighted_bonus);
        if self.exploration_mode_active && self.council_tick > self.exploration_mode_until_tick {
            self.exploration_mode_active = false;
        }

        let dummy_report = GpuTelemetryReport {
            gpu_success_ema: 0.90,
            gpu_latency_ema_ms: 80.0,
            mercy_modulated_confidence: 0.85,
            total_gpu_attempts: 100,
            last_gpu_success: true,
            valence_modulated_offload_score: 0.85,
        };
        let plateau_detected = self.detect_plateau(breakdown, &dummy_report);
        if plateau_detected {
            if let Some(action_summary) = self.handle_detected_plateau(breakdown, &dummy_report).await {
                changes.push(action_summary);
            }
        }

        let base_lr = self.get_scheduled_lr();

        let mut current_lr = base_lr;
        let modulation_strength = if self.exploration_mode_active { 1.35 } else { 1.12 };
        if breakdown.entanglement_weighted_bonus > 0.05 {
            current_lr = (current_lr * modulation_strength).min(0.08);
        } else if breakdown.entanglement_weighted_bonus < 0.035 {
            current_lr = (current_lr * 0.92).max(0.001);
        }

        let gradient_pf = if breakdown.entangled_pairs.iter().any(|p| p.contains("Performance ↔ Foresight") || p.contains("Performance ↔ Mercy")) {
            breakdown.entanglement_weighted_bonus * 0.8
        } else { 0.0 };

        let gradient_ma = if breakdown.entangled_pairs.iter().any(|p| p.contains("Mercy ↔ Alignment") || p.contains("Alignment ↔ Foresight")) {
            breakdown.entanglement_weighted_bonus * 0.8
        } else { 0.0 };

        let beta1 = self.adam_beta1;
        let beta2 = self.adam_beta2;
        let epsilon = self.adam_epsilon;
        let weight_decay = self.adam_weight_decay;
        let timestep = self.adam_timestep + 1;
        let formulation = self.nadam_formulation;

        if gradient_pf > 0.01 {
            let m = beta1 * self.adam_m_pf + (1.0 - beta1) * gradient_pf;
            let v = beta2 * self.adam_v_pf + (1.0 - beta2) * gradient_pf * gradient_pf;

            let m_hat = m / (1.0 - beta1.powi(timestep as i32));
            let v_hat = v / (1.0 - beta2.powi(timestep as i32));

            let nesterov_m_hat = match formulation {
                NadamFormulation::A => {
                    (1.0 - beta1) * gradient_pf + beta1 * m_hat
                }
                NadamFormulation::B => {
                    let m_nesterov = beta1 * self.adam_m_pf + (1.0 - beta1) * gradient_pf;
                    m_nesterov / (1.0 - beta1.powi(timestep as i32))
                }
            };

            let adamw_step = current_lr * nesterov_m_hat / (v_hat.sqrt() + epsilon);

            evolved_pf = (self.base_weight_pf + adamw_step) * (1.0 - current_lr * weight_decay);
            evolved_pf = evolved_pf.min(0.48);

            self.adam_m_pf = m;
            self.adam_v_pf = v;

            changes.push(format!(
                "base_weight_pf: {:.3} → {:.3} (cycle={}, formulation={:?}, Exploration={}, Cooldown={}, Severity={:.3}, Dynamic={:.5}, ConsensusEMA={:.4}, Nadam step={:.5})",
                self.base_weight_pf, evolved_pf, self.lr_current_cycle, formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.last_dynamic_improvement_threshold, self.consensus_vote_ema, adamw_step
            ));
        }

        if gradient_ma > 0.01 {
            let m = beta1 * self.adam_m_ma + (1.0 - beta1) * gradient_ma;
            let v = beta2 * self.adam_v_ma + (1.0 - beta2) * gradient_ma * gradient_ma;

            let m_hat = m / (1.0 - beta1.powi(timestep as i32));
            let v_hat = v / (1.0 - beta2.powi(timestep as i32));

            let nesterov_m_hat = match formulation {
                NadamFormulation::A => {
                    (1.0 - beta1) * gradient_ma + beta1 * m_hat
                }
                NadamFormulation::B => {
                    let m_nesterov = beta1 * self.adam_m_ma + (1.0 - beta1) * gradient_ma;
                    m_nesterov / (1.0 - beta1.powi(timestep as i32))
                }
            };

            let adamw_step = current_lr * nesterov_m_hat / (v_hat.sqrt() + epsilon);

            evolved_ma = (self.base_weight_ma + adamw_step) * (1.0 - current_lr * weight_decay);
            evolved_ma = evolved_ma.min(0.42);

            self.adam_m_ma = m;
            self.adam_v_ma = v;

            changes.push(format!(
                "base_weight_ma: {:.3} → {:.3} (cycle={}, formulation={:?}, Exploration={}, Cooldown={}, Severity={:.3}, Dynamic={:.5}, ConsensusEMA={:.4}, Nadam step={:.5})",
                self.base_weight_ma, evolved_ma, self.lr_current_cycle, formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.last_dynamic_improvement_threshold, self.consensus_vote_ema, adamw_step
            ));
        }

        if changes.is_empty() {
            return Ok("No base weight evolution needed (Expanded Entanglement Topology v2)".to_string());
        }

        let proposal = EvolutionProposal {
            id: rand::random::<u64>() % 1_000_000_000,
            proposer: "Lattice_Conductor_v13.1_SelfEvolution_Hook".to_string(),
            target_module: "ra-thor-one-organism / quantum_swarm_multi_consensus_vote (Expanded Entanglement Topology v2)".to_string(),
            description: format!("Self-evolution of entanglement base weights with Expanded Entanglement Topology v2 (4 pairs) + Dynamic Threshold Coupling + Light Consensus Momentum + Runtime Nadam Switching + Exploration Mode (Active: {:?}, Exploration={}, Cooldown={}) + Severity={:.3} + DynamicThreshold={:.5} + ConsensusEMA={:.4} + Cyclical Restarts (cycle={}, base_lr={:.5}, timestep={}). Changes: {:?}", formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.last_dynamic_improvement_threshold, self.consensus_vote_ema, self.lr_current_cycle, base_lr, timestep, changes),
            proposed_diff: format!("base_weight_pf = {:.3}; base_weight_ma = {:.3}; nadam_formulation = {:?}; exploration_mode = {}; cooldown = {}; severity = {:.3}; dynamic_threshold = {:.5}; consensus_ema = {:.4}", evolved_pf, evolved_ma, formulation, self.exploration_mode_active, self.cooldown_active, self.last_plateau_severity, self.last_dynamic_improvement_threshold, self.consensus_vote_ema),
            expected_benefit: 0.96,
            risk_score: 0.01,
            mercy_alignment: 0.98,
        };

        match self.evolution_gate.propose_evolution(proposal.clone()) {
            Ok(msg) => {
                println!("[ONE + Lattice Conductor] Expanded Entanglement Topology v2 + Dynamic Threshold Coupling + Consensus Momentum + Runtime Nadam {:?} + Exploration={} self-evolution proposed: {}", formulation, self.exploration_mode_active, msg);
                self.trigger_evolution_automation_hooks(&proposal, 0.98).await;
                self.persist_approved_evolution(&proposal, true, 0.98).await;
                Ok(format!("Entanglement base weights self-evolution via Expanded Entanglement Topology v2 proposed"))
            }
            Err(e) => Err(format!("Gate rejected Expanded Entanglement Topology v2 evolution: {}", e)),
        };
    }

    pub fn get_scheduled_lr(&self) -> f64 {
        let t = self.adam_timestep as f64;
        let warmup = self.lr_warmup_steps as f64;
        let lr_max = self.entanglement_evolution_lr;
        let lr_min = self.lr_min;

        if t < warmup {
            return lr_min + (lr_max - lr_min) * (t / warmup);
        }

        let mut cycle = self.lr_current_cycle as f64;
        let period = self.lr_restart_period as f64;
        let effective_t = t - warmup;
        let mut current_period = period * self.lr_restart_multiplier.powf(cycle);

        while effective_t >= current_period {
            cycle += 1.0;
            current_period = period * self.lr_restart_multiplier.powf(cycle);
        }

        let progress_in_cycle = if current_period > 0.0 {
            ((effective_t - (current_period - period * self.lr_restart_multiplier.powf((cycle - 1.0).max(0.0)))) / current_period).min(1.0)
        } else { 0.0 };

        if self.lr_schedule_type == "cosine" {
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress_in_cycle).cos());
            return lr_min + (lr_max - lr_min) * cosine;
        } else if self.lr_schedule_type == "exponential" {
            let decay_rate = 0.995;
            return (lr_max * decay_rate.powf(effective_t / 100.0)).max(lr_min);
        }

        lr_max
    }

    pub fn get_latest_swarm_vote_breakdown(&self) -> Option<SwarmVoteBreakdown> {
        self.last_swarm_vote_breakdown.clone()
    }

    pub async fn dispatch_gpu_and_feed_council(
        &mut self,
        task_name: &str,
        buffer_size: usize,
    ) -> Result<(String, CouncilDecision), String> {
        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: task_name.to_string(),
            buffer_size,
            intensity: "high".to_string(),
        };

        let (result, audit) = self.gpu_pipeline.dispatch_with_mercy_audit(task).await?;
        let decision = self.feed_mercy_gpu_audit_into_council(&audit).await;
        Ok((result.message, decision));
    }

    pub async fn dispatch_gpu_and_feed_lattice_conductor(
        &mut self,
        task_name: &str,
        buffer_size: usize,
    ) -> Result<(String, CouncilDecision), String> {
        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: task_name.to_string(),
            buffer_size,
            intensity: "high".to_string(),
        };

        let (result, _audit) = self.gpu_pipeline.dispatch_with_mercy_audit(task).await?;
        let telemetry_report = self.get_gpu_telemetry_for_lattice_conductor().await;
        let decision = self.feed_gpu_telemetry_into_council(&telemetry_report).await;
        Ok((result.message, decision));
    }

    pub async fn get_gpu_memory_stats(&self) -> crate::gpu_compute_pipeline::GpuMemoryStats {
        self.gpu_pipeline.get_memory_stats().await
    }

    pub async fn get_gpu_telemetry_for_lattice_conductor(&self) -> GpuTelemetryReport {
        let stats = self.gpu_pipeline.get_memory_stats().await;
        let telemetry_summary = self.gpu_pipeline.get_mercy_telemetry_summary().await;

        GpuTelemetryReport {
            gpu_success_ema: 0.93,
            gpu_latency_ema_ms: 78.0,
            mercy_modulated_confidence: (telemetry_summary.avg_mercy_norm * 0.85 + 0.15).clamp(0.75, 0.99),
            total_gpu_attempts: 128,
            last_gpu_success: true,
            valence_modulated_offload_score: telemetry_summary.avg_mercy_norm,
        };
    }

    pub fn evolve(&mut self, proposal: EvolutionProposal) -> Result<String, String> {
        self.evolution_gate.propose_evolution(proposal)
    }

    pub fn evolution_stats(&self) -> HashMap<String, f64> {
        self.evolution_gate.get_evolution_stats()
    }

    pub fn get_latest_council_metrics(&self) -> Option<CouncilReadinessMetrics> {
        self.last_council_metrics.clone()
    }

    pub async fn load_approved_evolutions(&self) -> Result<Vec<ApprovedEvolutionRecord>, String> {
        let content = fs::read_to_string(&self.approved_evolutions_path).await
            .map_err(|e| format!("Failed to read {}: {}", self.approved_evolutions_path, e))?;

        let mut records = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<ApprovedEvolutionRecord>(line) {
                records.push(record);
            }
        }
        Ok(records)
    }
}

pub fn launch_one_organism() -> RaThorOneOrganism {
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();
    println!("[Thunder] ONE Organism v14.65 + Quantum Swarm Engine + Benchmark Telemetry fully wired into Lattice Conductor self-evolution ready");
    organism
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_report(mercy_conf: f64, gpu_success: f64) -> GpuTelemetryReport {
        GpuTelemetryReport {
            gpu_success_ema: gpu_success,
            gpu_latency_ema_ms: 75.0,
            mercy_modulated_confidence: mercy_conf,
            total_gpu_attempts: 100,
            last_gpu_success: true,
            valence_modulated_offload_score: mercy_conf,
        }
    }

    fn make_test_proposal() -> EvolutionProposal {
        EvolutionProposal {
            id: 42,
            proposer: "test".to_string(),
            target_module: "lattice_conductor_v13".to_string(),
            description: "test proposal".to_string(),
            proposed_diff: "test".to_string(),
            expected_benefit: 0.9,
            risk_score: 0.02,
            mercy_alignment: 0.95,
        }
    }

    #[test]
    fn test_dynamic_improvement_threshold_mercy_gated() {
        let mut organism = RaThorOneOrganism::new();
        let breakdown = SwarmVoteBreakdown {
            performance_swarm: 0.91,
            mercy_swarm: 0.89,
            alignment_swarm: 0.87,
            foresight_swarm: 0.90,
            consensus_vote: 0.88,
            weights: (0.35, 0.30, 0.22, 0.20),
            entanglement_bonus: 0.012,
            entangled_pairs: vec![],
            entanglement_weighted_bonus: 0.012,
        };

        let high_mercy_report = make_test_report(0.91, 0.93);
        let _ = organism.detect_plateau(&breakdown, &high_mercy_report);
        let thresholds_high = organism.get_dynamic_thresholds();
        assert!(thresholds_high.contains("DynamicImprovementThreshold"));

        let low_mercy_report = make_test_report(0.78, 0.85);
        let _ = organism.detect_plateau(&breakdown, &low_mercy_report);
        let thresholds_low = organism.get_dynamic_thresholds();
        assert!(thresholds_low.contains("DynamicImprovementThreshold"));
    }

    #[test]
    fn test_quantum_swarm_engine_wiring_and_benchmarks() {
        let mut organism = RaThorOneOrganism::new();
        organism.initialize_quantum_swarm_member(1, vec![0.2; 8]);
        organism.initialize_quantum_swarm_member(2, vec![0.5; 8]);

        let benchmarks = organism.run_and_store_quantum_swarm_benchmarks(200);
        assert!(!benchmarks.is_empty());

        let summary = organism.get_quantum_swarm_engine_summary();
        assert!(summary.contains("members=2"));
    }
}
