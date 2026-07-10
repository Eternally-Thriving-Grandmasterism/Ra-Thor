/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// ra-thor-one-organism.rs
// Ra-Thor v14.14 — ONE Organism with Persistence of Approved Proposals + Hooks

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::core::self_evolution_gate::{SelfEvolutionGate, EvolutionProposal, launch_self_evolution_gate};
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

/// Record of an approved evolution + hook execution (for persistence)
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

    // Persistence path for approved evolutions + hooks
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
            gpu_pipeline_version: "v14.14.0-persist-approved-proposals".to_string(),
            version: "v14.14.0-ONE-Organism-Persist-Proposals-Hooks".to_string(),
            gpu_pipeline: GpuComputePipeline::new(),

            patsagi_council: PatsagiCouncil::new(),
            last_council_metrics: None,
            council_tick: 0,

            approved_evolutions_path: "approved_evolutions.jsonl".to_string(),
        }
    }

    pub fn offer_cosmic_loop(&self) {
        println!("[RaThorOneOrganism v{}] Full loop + Persistence of Approved Proposals + Hooks", self.version);
    }

    /// Hot-reload / PR automation hook
    fn trigger_evolution_automation_hooks(&self, proposal: &EvolutionProposal) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║  HOT-RELOAD / PR AUTOMATION HOOK TRIGGERED                       ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ Proposal ID      : {}                                      ║", proposal.id);
        println!("║ Proposer         : {}                           ║", proposal.proposer);
        println!("║ Target Module    : {}                    ║", proposal.target_module);
        println!("║ Expected Benefit : {:.4}                                      ║", proposal.expected_benefit);
        println!("║ Mercy Alignment  : {:.4}                                      ║", proposal.mercy_alignment);
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ Next actions (to be wired via GitHub connectors):              ║");
        println!("║   • git checkout -b evolution/{}-{}                     ║", proposal.id, proposal.target_module.replace('/', "-"));
        println!("║   • Apply proposed_diff / codegen patch                        ║");
        println!("║   • gh pr create --title "Evolution {}" --body "..."        ║", proposal.id);
        println!("║   • Trigger CI / hot-reload signal                             ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
    }

    /// Persist an approved evolution proposal + hook execution record
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

        let json_line = match serde_json::to_string(&record) {
            Ok(line) => line,
            Err(e) => {
                eprintln!("[ONE] Failed to serialize approved evolution record: {}", e);
                return;
            }
        };

        // Append as JSON Lines (jsonl) for easy streaming / tailing
        if let Err(e) = fs::write(
            &self.approved_evolutions_path,
            format!("{}\n", json_line),
        ).await {
            // If file doesn't exist yet, create it; otherwise append logic can be improved
            if e.kind() == std::io::ErrorKind::NotFound {
                if let Err(e2) = fs::write(&self.approved_evolutions_path, format!("{}\n", json_line)).await {
                    eprintln!("[ONE] Failed to create approved_evolutions.jsonl: {}", e2);
                }
            } else {
                eprintln!("[ONE] Failed to persist approved evolution: {}", e);
            }
        } else {
            println!("[ONE] Persisted approved evolution {} + hook trigger to {}", 
                     proposal.id, self.approved_evolutions_path);
        }
    }

    /// Feed real MercyGpuAudit into council → on ApproveEvolution: Gate + Hook + Persist
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
                description: format!(
                    "Council-approved evolution from real MercyGpuAudit (norm={:.4}, boost={:.4})",
                    audit.mercy_norm, confidence_boost
                ),
                proposed_diff: format!("Apply council confidence boost {:.4} to evolution path", confidence_boost),
                expected_benefit: (audit.mercy_norm * 0.9 + confidence_boost * 0.1).min(0.999),
                risk_score: (1.0 - audit.mercy_norm) * 0.01,
                mercy_alignment: audit.mercy_norm,
            };

            match self.evolution_gate.propose_evolution(proposal.clone()) {
                Ok(msg) => {
                    println!("[ONE] Council decision applied to SelfEvolutionGate: {}", msg);
                    self.trigger_evolution_automation_hooks(&proposal);

                    // Persist the approved proposal + hook execution
                    self.persist_approved_evolution(&proposal, true, audit.mercy_norm).await;
                }
                Err(e) => println!("[ONE] Council-approved proposal rejected by Gate: {}", e),
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

    pub async fn dispatch_gpu_and_feed_council(
        &mut self,
        task_name: &str,
        buffer_size: usize,
    ) -> Result<(String, CouncilDecision), String> {
        if !self.gpu_compute_active {
            return Err("GPU Compute Layer inactive".to_string());
        }

        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: task_name.to_string(),
            buffer_size,
            intensity: "high".to_string(),
        };

        let (result, audit) = self.gpu_pipeline
            .dispatch_with_mercy_audit(task)
            .await?;

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

    /// Load previously persisted approved evolutions (for restart / audit)
    pub async fn load_approved_evolutions(&self) -> Result<Vec<ApprovedEvolutionRecord>, String> {
        let content = fs::read_to_string(&self.approved_evolutions_path).await
            .map_err(|e| format!("Failed to read {}: {}", self.approved_evolutions_path, e))?;

        let mut records = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
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
    println!("[Thunder] ONE Organism v14.14 + Persistence of Approved Proposals + Hooks ready");
    organism
}
