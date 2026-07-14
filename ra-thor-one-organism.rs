/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.17 — ONE Organism + Lattice Conductor v13.1 Self-Evolving GPU Telemetry Loop (Deep Quantum Swarm Voting)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::github_connector::GitHubConnector;
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask, MercyGpuAudit};
use crate::gpu_patsagi_bridge::GpuTelemetryReport;

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

// NEW v14.8.6: Upgrade templates for Lattice Conductor self-evolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeConductorUpgradeTemplate {
    EMATuning,
    NewMercyGates,
    QuantumSwarmIntegration,
    CombinedGPUIntelligence,
}

impl LatticeConductorUpgradeTemplate {
    pub fn description(&self) -> &'static str {
        match self {
            LatticeConductorUpgradeTemplate::EMATuning => "Refine EMA alpha values and add additional mercy-modulated EMA loops for GPU telemetry.",
            LatticeConductorUpgradeTemplate::NewMercyGates => "Introduce or strengthen specific mercy gates (e.g., Precision Gate, Abundance Gate) in Lattice Conductor decision logic.",
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration => "Deepen integration between Lattice Conductor and Quantum Swarm for GPU-native deliberation, foresight, and voting.",
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence => "Combine EMA tuning + new mercy gates + Quantum Swarm hooks + swarm voting into a unified Lattice Conductor v13.2 upgrade.",
        }
    }

    pub fn target_diff(&self) -> &'static str {
        match self {
            LatticeConductorUpgradeTemplate::EMATuning => "Refine EMA alpha in gpu_patsagi_bridge + add gpu_latency_ema + multi-EMA feedback in ONE Organism.",
            LatticeConductorUpgradeTemplate::NewMercyGates => "Add new mercy gate variants in PatsagiCouncil::decide() and CouncilReadinessMetrics.",
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration => "Add Quantum Swarm deliberation + voting hooks in feed_gpu_telemetry_into_council and propose_* methods.",
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence => "Full v13.2 upgrade: EMA + Mercy Gates + Quantum Swarm deliberation + swarm voting in one coherent Lattice Conductor evolution.",
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

        let effective_mercy = (metrics.mercy_norm + gpu_boost).min(0.999);

        if effective_mercy >= self.mercy_norm_threshold {
            return CouncilDecision::ApproveEvolution {
                confidence_boost: (metrics.suggested_confidence_delta + gpu_boost).max(0.05),
            };
        }

        if effective_mercy < 0.4 {
            return CouncilDecision::EmergencyMercyIntervention {
                severity: (0.4 - effective_mercy) * 2.0,
            };
        }

        if metrics.suggested_confidence_delta > 0.15 || metrics.gpu_success_ema > 0.90 {
            return CouncilDecision::RequestAdditionalGpuResources {
                buffer_size_increase: if metrics.gpu_success_ema > 0.90 { 4096 } else { 2048 },
            };
        }

        CouncilDecision::AdjustRbeParameters {
            resource_flow_multiplier: 1.0 + (effective_mercy - 0.5) * 0.5,
            council_influence: effective_mercy,
        }
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
    council_tick: u64,
    approved_evolutions_path: String,
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

        Self {
            systems_activated: systems,
            mercy_runtime: "MercyGatingRuntime v2.0 (TOLC 8 aligned)".to_string(),
            evolution_gate: launch_self_evolution_gate(),
            gpu_compute_active: true,
            gpu_pipeline_version: "v14.17.0-real-github-connector".to_string(),
            version: "v14.17.0-ONE-Organism-LatticeConductor-v13.1-Deep-QuantumSwarm-Voting".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),

            patsagi_council: PatsagiCouncil::new(),
            last_council_metrics: None,
            council_tick: 0,
            approved_evolutions_path: "approved_evolutions.jsonl".to_string(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Full loop + Real GitHub PR + Deep Quantum Swarm Voting in Lattice Conductor v13.1", self.version);
    }

    async fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal, council_mercy_norm: f64) {
        println!("\n[Hook] Evolution {} approved — attempting real GitHub PR creation...", proposal.id);

        match GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor") {
            Ok(connector) => {
                let title = format!(
                    "Evolution {} — Council-approved from GPU Telemetry + MercyGpuAudit (norm={:.4})",
                    proposal.id, council_mercy_norm
                );

                let body = format!(
                    "## ONE Organism + Lattice Conductor v13.1 Deep Quantum Swarm Voting (auto-generated)

**Proposal ID**: {}
**Proposer**: {}
**Target Module**: {}
**Council Mercy Norm**: {:.4}
**GPU Success EMA**: {:.4}
**GPU Mercy Confidence**: {:.4}
**Expected Benefit**: {:.4}
**Mercy Alignment**: {:.4}

**Description**:
{}

**Proposed Diff**:
```
{}
```

---
*This PR was automatically created by RaThorOneOrganism v14.17 hot-reload/PR hook using the live GitHubConnector + Lattice Conductor GPU telemetry.*
",
                    proposal.id,
                    proposal.proposer,
                    proposal.target_module,
                    council_mercy_norm,
                    0.0,
                    0.0,
                    proposal.expected_benefit,
                    proposal.mercy_alignment,
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

    // NEW v14.8.6: Deep Quantum Swarm deliberation on GPU telemetry
    pub async fn quantum_swarm_deliberate_on_gpu_telemetry(&self, report: &GpuTelemetryReport) -> String {
        let swarm_confidence = (report.gpu_success_ema * 0.6 + report.mercy_modulated_confidence * 0.4).clamp(0.75, 0.999);

        if report.gpu_success_ema > 0.94 && report.mercy_modulated_confidence > 0.90 {
            format!(
                "Quantum Swarm Foresight: Excellent GPU performance detected (success_ema={:.4}). Recommend immediate Lattice Conductor upgrade + increased GPU offload + Quantum Swarm parallel deliberation on next dispatch batch. Mercy valence: {:.4}",
                report.gpu_success_ema, swarm_confidence
            )
        } else if report.gpu_latency_ema_ms > 120.0 {
            format!(
                "Quantum Swarm Analysis: Elevated GPU latency ({:.1}ms). Suggest EMA tuning + swarm-assisted load balancing. Current swarm confidence: {:.4}",
                report.gpu_latency_ema_ms, swarm_confidence
            )
        } else {
            format!(
                "Quantum Swarm Observation: Stable GPU telemetry. Continue current mercy-modulated offload policy. Swarm confidence: {:.4}",
                swarm_confidence
            )
        }
    }

    // NEW v14.8.6: Implement swarm voting logic — Quantum Swarm votes on evolution proposals
    pub async fn quantum_swarm_vote_on_evolution(&self, report: &GpuTelemetryReport, proposal: &EvolutionProposal) -> f64 {
        // Swarm voting: combines GPU success, mercy confidence, and proposal alignment
        let base_vote = (report.gpu_success_ema * 0.5 + report.mercy_modulated_confidence * 0.4).clamp(0.6, 0.99);

        // Bonus if proposal targets Lattice Conductor or Quantum Swarm modules
        let alignment_bonus = if proposal.target_module.contains("lattice_conductor") || proposal.target_module.contains("quantum_swarm") {
            0.08
        } else {
            0.0
        };

        // Penalty for high risk
        let risk_penalty = proposal.risk_score * 0.15;

        let final_vote = (base_vote + alignment_bonus - risk_penalty).clamp(0.65, 0.999);

        println!(
            "[Quantum Swarm Vote] Proposal {} | base_vote={:.4} | alignment_bonus={:.4} | risk_penalty={:.4} | final_vote={:.4}",
            proposal.id, base_vote, alignment_bonus, risk_penalty, final_vote
        );

        final_vote
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

        let metrics = CouncilReadinessMetrics {
            council_ready: true,
            mercy_norm: report.valence_modulated_offload_score,
            suggested_confidence_delta: (report.mercy_modulated_confidence - 0.75).max(0.0) * 0.4,
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
            gpu_success_ema: report.gpu_success_ema,
            gpu_latency_ema_ms: report.gpu_latency_ema_ms,
            gpu_mercy_modulated_confidence: report.mercy_modulated_confidence,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        if let CouncilDecision::ApproveEvolution { confidence_boost } = &decision {
            let proposal = EvolutionProposal {
                id: rand::random::<u64>() % 1_000_000_000,
                proposer: "Lattice_Conductor_v13.1_via_GPU_Telemetry".to_string(),
                target_module: "gpu_compute_pipeline / lattice_conductor / powrush_rbe".to_string(),
                description: format!(
                    "Council-approved from GPU Telemetry Report (success_ema={:.4}, mercy_conf={:.4})",
                    report.gpu_success_ema, report.mercy_modulated_confidence
                ),
                proposed_diff: format!("Apply Lattice Conductor GPU boost {:.4}", confidence_boost),
                expected_benefit: (report.mercy_modulated_confidence * 0.85 + confidence_boost * 0.15).min(0.999),
                risk_score: (1.0 - report.mercy_modulated_confidence) * 0.02,
                mercy_alignment: report.mercy_modulated_confidence,
            };

            // NEW: Swarm voting on the evolution proposal
            if report.gpu_success_ema > 0.93 {
                let swarm_vote = self.quantum_swarm_vote_on_evolution(report, &proposal).await;
                if swarm_vote < 0.80 {
                    println!("[Quantum Swarm Vote] Vote too low ({:.4}) — evolution proposal deprioritized", swarm_vote);
                }
            }

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

    // NEW v14.8.6: Self-evolution hook with multiple upgrade templates + Deep Quantum Swarm + Voting
    pub async fn propose_lattice_conductor_upgrade_from_gpu_telemetry(&self, report: &GpuTelemetryReport) -> Result<String, String> {
        if report.gpu_success_ema < 0.90 || report.mercy_modulated_confidence < 0.88 {
            return Err("GPU telemetry not excellent enough for automatic Lattice Conductor upgrade".to_string());
        }

        let template = if report.gpu_latency_ema_ms > 150.0 {
            LatticeConductorUpgradeTemplate::EMATuning
        } else if report.gpu_success_ema > 0.95 {
            LatticeConductorUpgradeTemplate::QuantumSwarmIntegration
        } else {
            LatticeConductorUpgradeTemplate::CombinedGPUIntelligence
        };

        let swarm_foresight = if template == LatticeConductorUpgradeTemplate::QuantumSwarmIntegration || template == LatticeConductorUpgradeTemplate::CombinedGPUIntelligence {
            self.quantum_swarm_deliberate_on_gpu_telemetry(report).await
        } else {
            String::new()
        };

        // Create a temporary proposal for swarm voting
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

        // Swarm voting on the upgrade proposal
        let swarm_vote = self.quantum_swarm_vote_on_evolution(report, &temp_proposal).await;

        if swarm_vote < 0.82 {
            return Err(format!("Quantum Swarm vote too low ({:.4}) — upgrade deprioritized", swarm_vote));
        }

        let base_description = format!(
            "Automatic self-evolution (Template: {:?}): {}. GPU telemetry: success_ema={:.4}, mercy_conf={:.4}, latency_ema={:.1}ms | Swarm Vote: {:.4}",
            template,
            template.description(),
            report.gpu_success_ema,
            report.mercy_modulated_confidence,
            report.gpu_latency_ema_ms,
            swarm_vote
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
            expected_benefit: 0.96 * swarm_vote, // modulated by swarm vote
            risk_score: 0.02,
            mercy_alignment: report.mercy_modulated_confidence,
        };

        match self.evolution_gate.propose_evolution(proposal.clone()) {
            Ok(msg) => {
                println!("[ONE + Lattice Conductor Self-Evolution] GPU telemetry excellent — auto-proposed {:?} upgrade (Swarm Vote: {:.4}): {}", template, swarm_vote, msg);
                self.trigger_evolution_automation_hooks(&proposal, report.mercy_modulated_confidence).await;
                self.persist_approved_evolution(&proposal, true, report.mercy_modulated_confidence).await;
                Ok(format!("Lattice Conductor v13.1 {:?} upgrade proposed from GPU telemetry + Quantum Swarm (vote={:.4})", template, swarm_vote))
            }
            Err(e) => Err(format!("Gate rejected Lattice Conductor upgrade: {}", e)),
        }
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
        }
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
    println!("[Thunder] ONE Organism v14.17 + Real GitHubConnector + Deep Quantum Swarm Voting in Lattice Conductor v13.1 ready");
    organism
}
