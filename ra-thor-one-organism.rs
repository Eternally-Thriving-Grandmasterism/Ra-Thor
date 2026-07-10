/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.17 — ONE Organism with Live GitHub PR Creation via github_connector

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
use crate::github_connector::GitHubConnector;
use crate::gpu_compute_pipeline::{GpuComputePipeline, GpuTask, MercyGpuAudit};

// === Council + Decision Types ===

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

        Self {
            systems_activated: systems,
            mercy_runtime: "MercyGatingRuntime v2.0 (TOLC 8 aligned)".to_string(),
            evolution_gate: launch_self_evolution_gate(),
            gpu_compute_active: true,
            gpu_pipeline_version: "v14.17.0-real-github-connector".to_string(),
            version: "v14.17.0-ONE-Organism-Real-GitHub-Connector".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),

            patsagi_council: PatsagiCouncil::new(),
            last_council_metrics: None,
            council_tick: 0,
            approved_evolutions_path: "approved_evolutions.jsonl".to_string(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Full loop + Real GitHub PR Creation via github_connector", self.version);
    }

    /// Live GitHub PR creation using the real github_connector module
    async fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal, council_mercy_norm: f64) {
        println!("\n[Hook] Evolution {} approved — attempting real GitHub PR creation...", proposal.id);

        match GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor") {
            Ok(connector) => {
                let title = format!(
                    "Evolution {} — Council-approved from real MercyGpuAudit (norm={:.4})",
                    proposal.id, council_mercy_norm
                );

                let body = format!(
                    "## ONE Organism Evolution (auto-generated)

**Proposal ID**: {}
**Proposer**: {}
**Target Module**: {}
**Council Mercy Norm**: {:.4}
**Expected Benefit**: {:.4}
**Mercy Alignment**: {:.4}

**Description**:
{}

**Proposed Diff**:
```
{}
```

---
*This PR was automatically created by RaThorOneOrganism v14.17 hot-reload/PR hook using the live GitHubConnector.*
",
                    proposal.id,
                    proposal.proposer,
                    proposal.target_module,
                    council_mercy_norm,
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

    pub async fn feed_mercy_gpu_audit_into_council(&mut self, audit: &MercyGpuAudit) -> CouncilDecision {
        self.council_tick += 1;

        let metrics = CouncilReadinessMetrics {
            council_ready: audit.council_ready,
            mercy_norm: audit.mercy_norm,
            suggested_confidence_delta: audit.suggested_confidence_delta(),
            evolution_level: self.evolution_stats().get("evolution_level").copied().unwrap_or(0.0) as u32,
            last_updated_tick: self.council_tick,
        };

        self.last_council_metrics = Some(metrics.clone());

        let decision = self.patsagi_council.decide(&metrics);

        if let CouncilDecision::ApproveEvolution { confidence_boost } = &decision {
            let proposal = EvolutionProposal {
                id: rand::random::<u64>() % 1_000_000_000,
                proposer: "PATSAGi_Council_via_GPU_Audit".to_string(),
                target_module: "gpu_compute_pipeline / powrush_rbe".to_string(),
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
        Ok((result.message, decision))
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
    println!("[Thunder] ONE Organism v14.17 + Real GitHubConnector PR creation ready");
    organism
}
