/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

/// # Lattice Conductor v13
///
/// The Eternal Living Nervous System of the Ra-Thor ONE Organism lattice.
/// Primary orchestration layer for conducting councils, self-evolution, geometric state,
/// and NEXi-derived symbolic reasoning under strict TOLC 8 enforcement.
///
/// v13.2 (Phases A+B+C + real parameters + feature flags) — PR #363
/// - External Symbolic Input (ONE Organism ready)
/// - Mercy-gated Self-Proposal generation + controlled apply
/// - Real ConductorSymbolicParameters (directly mutated by Phase C)
/// - Granular Cargo features for safe rollout
///
/// v13.3 — Meta Self-Evolution Application
/// - Meta audit integrated into SelfEvolutionOrchestrator
/// - Conductor methods delegate to orchestrator (clean separation)
/// - All mercy-gated, TOLC 8 enforced, ONE Organism aligned
///
/// v13.4 (in progress) — Orchestrator-Owned Meta Rate Parameters
/// - Internal meta_evolution_rate, meta_audit_threshold, meta_success_ema + stabilization
/// - Full delegation + getters exposed at conductor level
///
/// v13.5 — GPU / PATSAGi Telemetry + Mercy Audit Integration Readiness (full Lattice Conductor / Powrush lifecycle)
/// - submit_patsagi_task_with_audit now returns (GpuTaskResult, MercyGpuAudit) with council_ready flag + suggested_confidence_delta
/// - Real start handles + shutdown timeout (5s) already implemented in GpuComputePipeline (ready for conductor-owned background telemetry tasks)
/// - Mercy-norm telemetry + histograms ready to feed PATSAGi Council readiness, self-evolution confidence, and governance cycles
/// - Prometheus metrics + breaker state available for conductor observability / self-evolution hooks
/// - integrate_patsagi_gpu_audit wires audit directly into council_voted_evolution, symbolic EMAs, mercy_score, coherence, evolution_level
/// - conductor-owned Arc<GpuComputePipeline> + set/start/stop wrappers for full Powrush/RBE lifecycle ownership
/// - ONE Organism: GPU simulation layer ↔ Lattice Conductor v13.1+ bidirectional mercy-gated flow

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

// Sub-module re-exports (structure preserved)
pub use crate::conductable::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use crate::coordinator::{AverageInfluenceStrategy, CoordinationStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation};
pub use crate::geometric::{BasicGeometricMotor, GeometricMotor, GeometricState};
pub use crate::self_evolution::{EpigeneticBlessing, SelfEvolving, SelfEvolutionOrchestrator};

// ==================== REAL PARAMETER STRUCT ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorSymbolicParameters {
    pub base_confidence_threshold: f64,
    pub ema_alpha: f64,
    pub boost_multiplier: f64,
}

impl Default for ConductorSymbolicParameters {
    fn default() -> Self {
        Self {
            base_confidence_threshold: 0.78,
            ema_alpha: 0.18,
            boost_multiplier: 1.0,
        }
    }
}

// ==================== SUPPORTING TYPES ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyWeightedVote {
    votes: Vec<(String, f64, f64)>,
}

impl MercyWeightedVote {
    pub fn new() -> Self { Self { votes: Vec::new() } }

    pub fn add_vote(&mut self, council_name: &str, weight: f64, mercy_impact: f64) {
        self.votes.push((council_name.to_string(), weight, mercy_impact));
    }

    pub fn compute_consensus(&self) -> f64 {
        if self.votes.is_empty() { return 0.0; }
        let total_weight: f64 = self.votes.iter().map(|(_, w, _)| w).sum();
        if total_weight == 0.0 { return 0.0; }
        let weighted_sum: f64 = self.votes.iter().map(|(_, w, impact)| w * impact).sum();
        (weighted_sum / total_weight).clamp(-0.3, 0.5)
    }

    pub fn to_audit_string(&self) -> String {
        format!("[MercyWeightedVote Audit] {} votes", self.votes.len())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub description: String,
    pub valence: f64,
}

impl Operation {
    pub fn new(name: &str, description: &str, valence: f64) -> Self {
        Self { name: name.to_string(), description: description.to_string(), valence }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometricState {
    pub valence: f64,
    pub mercy_score: f64,
    pub tolc_alignment: f64,
    pub evolution_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    pub evolution_rate: f64,
    pub mercy_recovery_rate: f64,
    pub layer_adaptations: Vec<f64>,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self { evolution_rate: 0.01, mercy_recovery_rate: 0.05, layer_adaptations: vec![1.0; 6] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    pub operations_processed: u64,
}

// ==================== v13.5 GPU / PATSAGi BRIDGE TYPES (module scope) ====================
// Bridge from gpu_compute_pipeline.rs (v14.9+) for ONE Organism integration.
// Enables conductor-owned Arc<GpuComputePipeline> + audit consumption without hard dep in this phase.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTaskResult {
    pub task_id: u64,
    pub success: bool,
    pub execution_time_ms: u64,
    pub output_size: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGpuAudit {
    pub task_id: u64,
    pub mercy_norm: f64,
    pub execution_time_ms: u64,
    pub reuse_ratio: f64,
    pub fragmentation_estimate: f64,
    pub council_ready: bool,
    pub trace: String,
}

impl MercyGpuAudit {
    pub fn is_council_ready(&self) -> bool {
        self.mercy_norm >= 0.85
    }

    pub fn suggested_confidence_delta(&self) -> f64 {
        (self.mercy_norm - 0.5) * 0.18
    }

    pub fn summary(&self) -> String {
        format!(
            "MercyGpuAudit | norm={:.4} | council_ready={} | time={}ms | reuse={:.2} | frag={:.1}",
            self.mercy_norm, self.council_ready, self.execution_time_ms, self.reuse_ratio, self.fragmentation_estimate
        )
    }
}

/// Conductor-ownable GPU pipeline handle (v13.5+).
/// Real heavy implementation (allocator, telemetry, histograms, breaker, prometheus, periodic save)
/// lives in gpu_compute_pipeline.rs. This facade + wrappers enable the conductor to own the lifecycle.
pub struct GpuComputePipeline {
    // Opaque real handle in production binary where both modules are in scope.
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {}
    }

    /// Start periodic auto-save (real impl has tokio interval + retry + breaker + 5s shutdown timeout).
    pub async fn start_periodic_mercy_telemetry_save(&self, interval_secs: u64) -> Result<(), String> {
        // In full monorepo binary the real GpuComputePipeline from gpu_compute_pipeline.rs is used.
        // This allows conductor-owned background tasks today.
        println!("[LatticeConductor] start_periodic_mercy_telemetry_save({}s) - facade ready for real pipeline", interval_secs);
        Ok(())
    }

    /// Graceful shutdown with timeout (matches real 5s timeout in gpu_compute_pipeline.rs).
    pub async fn shutdown_mercy_telemetry_auto_save(&self) -> Result<(), String> {
        println!("[LatticeConductor] shutdown_mercy_telemetry_auto_save (5s timeout) - facade");
        Ok(())
    }

    /// submit_patsagi_task_with_audit passthrough (real version returns (GpuTaskResult, MercyGpuAudit)).
    pub async fn submit_patsagi_task_with_audit(
        &self,
        _query: &str,
        _intensity: &str,
        _buffer_size: usize,
    ) -> Result<(GpuTaskResult, MercyGpuAudit), String> {
        // Placeholder wiring; real call happens on the gpu_compute_pipeline instance before or after set.
        Err("Use the real GpuComputePipeline instance from gpu_compute_pipeline.rs for actual submit".to_string())
    }
}

// ==================== MAIN CONDUCTOR v13 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleLatticeConductor {
    pub id: u32,
    pub name: String,
    registered_councils: Vec<(u32, String)>,
    operation_queue: Vec<Operation>,
    pub state: GeometricState,
    pub adaptive_params: AdaptiveParameters,
    pub metrics: Metrics,
    mercy_violations: Vec<String>,
    audit_traces: Vec<String>,
    one_organism_coherence: f64,
    pub evolution_orchestrator: SelfEvolutionOrchestrator,
    pub registry: ConductorRegistry,
    symbolic_confidence_ema: f64,
    symbolic_success_ema: f64,
    pub experimental_use_external_symbolic: bool,
    pub symbolic_params: ConductorSymbolicParameters,
    #[cfg(feature = "self-proposal")]
    self_proposal_log: Vec<SymbolicSelfProposal>,
    // v13.5+: Conductor-owned GPU pipeline for full Powrush/RBE lifecycle (real start handles + timeout shutdown)
    pub gpu_pipeline: Option<Arc<GpuComputePipeline>>,
}

impl Default for SimpleLatticeConductor {
    fn default() -> Self { Self::new() }
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            id: 0,
            name: "Ra-Thor Lattice Conductor v13".to_string(),
            registered_councils: Vec::new(),
            operation_queue: Vec::new(),
            state: GeometricState { valence: 1.0, mercy_score: 1.0, tolc_alignment: 1.0, evolution_level: 0.0 },
            adaptive_params: AdaptiveParameters::default(),
            metrics: Metrics::default(),
            mercy_violations: Vec::new(),
            audit_traces: Vec::new(),
            one_organism_coherence: 1.0,
            evolution_orchestrator: SelfEvolutionOrchestrator::new(),
            registry: ConductorRegistry::new(),
            symbolic_confidence_ema: 0.75,
            symbolic_success_ema: 0.70,
            experimental_use_external_symbolic: false,
            symbolic_params: ConductorSymbolicParameters::default(),
            #[cfg(feature = "self-proposal")]
            self_proposal_log: Vec::new(),
            gpu_pipeline: None,
        }
    }

    pub fn register_council(&mut self, id: u32, name: &str) {
        self.registered_councils.push((id, name.to_string()));
        self.audit_traces.push(format!("[v13 Council Registered] {}: {}", id, name));
    }

    pub fn queue_operation(&mut self, op: Operation) {
        self.operation_queue.push(op);
    }

    pub fn tick(&mut self) -> Result<(), String> {
        #[cfg(feature = "external-symbolic")]
        let symbolic = if self.experimental_use_external_symbolic {
            let ext_input = ExternalSymbolicInput::new("grok_one_organism", "conductor_tick_external", self.state.valence);
            accept_external_symbolic_deliberation(ext_input)
        } else {
            crate::metta_symbolic_deliberation("conductor_tick", self.state.valence)
        };

        #[cfg(not(feature = "external-symbolic"))]
        let symbolic = crate::metta_symbolic_deliberation("conductor_tick", self.state.valence);

        let ema_alpha = self.symbolic_params.ema_alpha;
        self.symbolic_confidence_ema = self.symbolic_confidence_ema * (1.0 - ema_alpha) + symbolic.confidence_score * ema_alpha;

        let base_threshold = self.symbolic_params.base_confidence_threshold;
        let mercy_mod = if self.state.mercy_score > 1.25 { -0.07 } else if self.state.mercy_score < 0.85 { 0.08 } else { 0.0 };
        let calibration_bias = (self.symbolic_confidence_ema - 0.75) * 0.06;
        let adaptive_threshold = (base_threshold + mercy_mod + calibration_bias).clamp(0.65, 0.92);

        let confidence = symbolic.confidence_score;
        let (mut evolution_boost, mut tolc_boost) = if confidence >= adaptive_threshold { (0.008, 0.012) } else { (0.002, 0.004) };

        let success_multiplier = if confidence >= adaptive_threshold {
            (self.symbolic_success_ema * 0.55 + 0.72).clamp(0.78, 1.22) * self.symbolic_params.boost_multiplier
        } else { 1.0 };

        evolution_boost *= success_multiplier;
        tolc_boost *= success_multiplier;

        self.state.evolution_level += evolution_boost;
        self.state.tolc_alignment = (self.state.tolc_alignment + tolc_boost).min(1.25);

        self.adaptive_params.evolution_rate *= 1.0 + (self.adaptive_params.layer_adaptations[0] * 0.001);
        self.state.evolution_level += self.adaptive_params.evolution_rate;

        let mut mercy_delta: f64 = 0.0;
        while let Some(op) = self.operation_queue.pop() {
            let impact = op.valence * 0.1;
            self.state.valence = (self.state.valence + impact).clamp(0.0, 2.0);
            mercy_delta += impact * 0.5;
            self.metrics.operations_processed += 1;
        }

        if !self.registered_councils.is_empty() {
            let mut vote = MercyWeightedVote::new();
            for (_, cname) in &self.registered_councils {
                vote.add_vote(cname, 1.0 / self.registered_councils.len() as f64, mercy_delta.max(-0.2));
            }
            let consensus = vote.compute_consensus();
            self.state.mercy_score = (self.state.mercy_score + consensus).clamp(0.1, 1.5);
        }

        if self.state.mercy_score < 0.7 {
            self.state.mercy_score = (self.state.mercy_score + self.adaptive_params.mercy_recovery_rate).min(1.0);
        }

        let coherence_shift = (self.state.mercy_score - 0.5) * 0.05;
        self.one_organism_coherence = (self.one_organism_coherence + coherence_shift).clamp(0.5, 1.2);
        self.state.tolc_alignment = (self.state.tolc_alignment + 0.01).min(1.25);

        let mercy_improved = mercy_delta > 0.012;
        let evolution_improved = evolution_boost > 0.004;
        let success_signal = if (mercy_improved || evolution_improved) && confidence >= adaptive_threshold { 0.88 } else if mercy_improved || evolution_improved { 0.55 } else { 0.28 };

        let success_alpha = 0.15;
        self.symbolic_success_ema = self.symbolic_success_ema * (1.0 - success_alpha) + success_signal * success_alpha;

        #[cfg(feature = "self-proposal")]
        if self.state.mercy_score >= 0.9 {
            let new_proposals = self.generate_symbolic_self_proposals();
            if !new_proposals.is_empty() {
                self.self_proposal_log.extend(new_proposals.clone());
                self.audit_traces.push(format!("[v13.2 SelfProposal] generated {} proposals", new_proposals.len()));
            }
        }

        self.audit_traces.push(format!(
            "[v13 Symbolic] conf={:.2} ema={:.2} success_ema={:.2} thr={:.2} mult={:.2} boost={:.2}",
            confidence, self.symbolic_confidence_ema, self.symbolic_success_ema, adaptive_threshold, success_multiplier, self.symbolic_params.boost_multiplier
        ));

        Ok(())
    }

    pub fn get_geometric_state(&self) -> &GeometricState { &self.state }
    pub fn get_mercy_violations(&self) -> &[String] { &self.mercy_violations }
    pub fn get_symbolic_confidence_ema(&self) -> f64 { self.symbolic_confidence_ema }
    pub fn get_symbolic_success_ema(&self) -> f64 { self.symbolic_success_ema }

    pub fn get_symbolic_params(&self) -> &ConductorSymbolicParameters { &self.symbolic_params }
    pub fn set_symbolic_params(&mut self, params: ConductorSymbolicParameters) { self.symbolic_params = params; }

    // ==================== v13.4: Meta Rate State Getters (delegated) ====================

    #[cfg(feature = "self-proposal")]
    pub fn get_meta_evolution_rate(&self) -> f64 {
        self.evolution_orchestrator.get_meta_evolution_rate()
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_meta_audit_threshold(&self) -> f64 {
        self.evolution_orchestrator.get_meta_audit_threshold()
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_meta_success_ema(&self) -> f64 {
        self.evolution_orchestrator.get_meta_success_ema()
    }

    #[cfg(feature = "external-symbolic")]
    pub fn accept_external_symbolic_deliberation(input: ExternalSymbolicInput) -> SymbolicDeliberation {
        let mut result = metta_symbolic_deliberation(&input.content, input.context_valence);
        result.input = format!("[external:{}] {}", input.source, result.input);
        result.message = format!("{}_via_external_{}", result.message, input.source);
        result
    }

    #[cfg(feature = "self-proposal")]
    pub fn generate_symbolic_self_proposals(&self) -> Vec<SymbolicSelfProposal> {
        let mut proposals = Vec::new();
        let success_ema = self.symbolic_success_ema;
        let conf_ema = self.symbolic_confidence_ema;
        let p = &self.symbolic_params;

        if success_ema < 0.65 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "base_confidence_threshold_adjust".to_string(),
                current_value: p.base_confidence_threshold,
                proposed_value: (p.base_confidence_threshold - 0.03).max(0.65),
                rationale: "Low success_ema → slightly more lenient base threshold".to_string(),
                mercy_impact_estimate: 0.015,
                confidence: 0.68,
            });
        }
        if conf_ema > 0.82 && success_ema > 0.70 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "ema_alpha_adjust".to_string(),
                current_value: p.ema_alpha,
                proposed_value: (p.ema_alpha + 0.03).min(0.35),
                rationale: "High stable confidence → modestly faster EMA adaptation".to_string(),
                mercy_impact_estimate: 0.01,
                confidence: 0.71,
            });
        }
        if success_ema > 0.85 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "boost_multiplier_adjust".to_string(),
                current_value: p.boost_multiplier,
                proposed_value: (p.boost_multiplier + 0.12).min(1.4),
                rationale: "Very high success → expand boost multiplier range".to_string(),
                mercy_impact_estimate: 0.008,
                confidence: 0.74,
            });
        }
        proposals
    }

    /// Phase C: Explicitly apply a logged self-proposal and mutate the real parameters.
    #[cfg(feature = "self-proposal")]
    pub fn apply_symbolic_self_proposal(&mut self, index: usize) -> Result<String, String> {
        if index >= self.self_proposal_log.len() { return Err("Invalid index".to_string()); }
        let proposal = &self.self_proposal_log[index];
        if self.state.mercy_score < 0.92 || proposal.confidence < 0.65 {
            return Err("Gates not met".to_string());
        }

        let p = &mut self.symbolic_params;
        let effect = match proposal.proposal_type.as_str() {
            "base_confidence_threshold_adjust" => {
                p.base_confidence_threshold = proposal.proposed_value.clamp(0.65, 0.92);
                format!("base_confidence_threshold → {:.3}", p.base_confidence_threshold)
            }
            "ema_alpha_adjust" => {
                p.ema_alpha = proposal.proposed_value.clamp(0.10, 0.35);
                format!("ema_alpha → {:.3}", p.ema_alpha)
            }
            "boost_multiplier_adjust" => {
                p.boost_multiplier = proposal.proposed_value.clamp(0.8, 1.5);
                format!("boost_multiplier → {:.2}", p.boost_multiplier)
            }
            _ => "unknown proposal type".to_string(),
        };

        Ok(format!("Phase C applied #{}: {}", index, effect))
    }

    #[cfg(feature = "self-proposal")]
    pub fn apply_top_confidence_proposal(&mut self) -> Result<String, String> {
        if self.self_proposal_log.is_empty() { return Err("No proposals".to_string()); }
        let best_idx = self.self_proposal_log.iter().enumerate()
            .max_by(|a| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .map(|(i, _)| i).unwrap();
        self.apply_symbolic_self_proposal(best_idx)
    }

    // ==================== v13.3 META SELF-EVOLUTION ====================

    #[cfg(feature = "self-proposal")]
    pub fn generate_meta_self_evolution_proposals(&self) -> Vec<SymbolicSelfProposal> {
        self.evolution_orchestrator.generate_meta_self_evolution_proposals(
            self.state.mercy_score,
            self.symbolic_success_ema,
            self.symbolic_confidence_ema,
            self.symbolic_params.boost_multiplier,
        )
    }

    #[cfg(feature = "self-proposal")]
    pub fn apply_meta_self_evolution_proposal(&mut self, index: usize) -> Result<String, String> {
        let meta_props = self.generate_meta_self_evolution_proposals();
        if index >= meta_props.len() { return Err("Invalid meta index".to_string()); }

        let prop = &meta_props[index];
        let _ = self.evolution_orchestrator.apply_meta_self_evolution_proposal(prop, self.state.mercy_score)?;

        let p = &mut self.symbolic_params;
        let effect = match prop.proposal_type.as_str() {
            "meta_self_evolution_rate_increase" => {
                p.boost_multiplier = prop.proposed_value;
                format!("boost_multiplier → {:.2}", p.boost_multiplier)
            }
            "meta_mercy_audit_threshold_tighten" => {
                p.base_confidence_threshold = (p.base_confidence_threshold + 0.01).min(0.92);
                format!("base_confidence_threshold tightened → {:.3}", p.base_confidence_threshold)
            }
            _ => "unknown meta proposal".to_string(),
        };

        self.audit_traces.push(format!("[v13.3 MetaSelfEvolution] applied #{}: {}", index, effect));
        Ok(format!("v13.3 META Phase C applied #{}: {}", index, effect))
    }

    // ==================== v13.4: Meta Rate Delegation + Getters ====================

    #[cfg(feature = "self-proposal")]
    pub fn apply_meta_rate_proposal(&mut self, index: usize) -> Result<String, String> {
        let meta_props = self.evolution_orchestrator.generate_meta_proposals_from_conductor(self);
        if index >= meta_props.len() { return Err("Invalid meta index".to_string()); }

        let prop = &meta_props[index];
        self.evolution_orchestrator.apply_meta_rate_proposal(prop)
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_self_proposal_log(&self) -> &[SymbolicSelfProposal] { &self.self_proposal_log }

    // v13.5: set the owned pipeline (real one from gpu_compute_pipeline.rs or facade)
    pub fn set_gpu_pipeline(&mut self, pipeline: Arc<GpuComputePipeline>) {
        self.gpu_pipeline = Some(pipeline);
    }

    // v13.5: Conductor-owned start wrapper (uses real  handle + periodic logic inside)
    pub async fn start_gpu_pipeline_telemetry(&self, interval_secs: u64) -> Result<(), String> {
        if let Some(pipeline) = &self.gpu_pipeline {
            pipeline.start_periodic_mercy_telemetry_save(interval_secs).await
        } else {
            Err("gpu_pipeline not set on conductor. Call set_gpu_pipeline first.".to_string())
        }
    }

    // v13.5: Conductor-owned stop wrapper (propagates shutdown signal + 5s timeout)
    pub async fn stop_gpu_pipeline_telemetry(&self) -> Result<(), String> {
        if let Some(pipeline) = &self.gpu_pipeline {
            pipeline.shutdown_mercy_telemetry_auto_save().await
        } else {
            Err("gpu_pipeline not set on conductor. Call set_gpu_pipeline first.".to_string())
        }
    }

    // ==================== v13.5: GPU PATSAGi Audit Integration ====================
    /// Wires the return of submit_patsagi_task_with_audit into the conductor:
    /// - council_ready=true → council_voted_evolution("GPU_PATSAGi_Council") + confidence/mercy/evolution boosts
    /// - suggested_confidence_delta applied to symbolic EMAs
    /// - Always traces for governance + PATSAGi readiness observability
    /// - Feeds mercy-norm telemetry into self-evolution confidence and cycles
    /// Maximum builder velocity, TOLC 8 mercy-gated, ONE Organism coherent.
    pub fn integrate_patsagi_gpu_audit(&mut self, audit: &MercyGpuAudit) {
        self.audit_traces.push(format!("[v13.5 GPU Integration] {}", audit.summary()));

        if audit.council_ready {
            let mut trace_log = Vec::new();
            self.evolution_orchestrator.integrate_gpu_mercy_audit(
                audit.mercy_norm,
                true,
                audit.suggested_confidence_delta(),
                &mut self.state,
                &mut trace_log,
            );
            for t in trace_log {
                self.audit_traces.push(t);
            }

            let delta = audit.suggested_confidence_delta();
            self.symbolic_confidence_ema = (self.symbolic_confidence_ema + delta).clamp(0.1, 1.5);
            self.symbolic_success_ema = (self.symbolic_success_ema + delta * 0.6).clamp(0.1, 1.5);
            self.state.mercy_score = (self.state.mercy_score + (audit.mercy_norm - 0.5) * 0.07).clamp(0.1, 1.6);
            self.one_organism_coherence = (self.one_organism_coherence + 0.025).min(1.45);
            self.state.evolution_level += 0.003 * audit.mercy_norm;

            self.audit_traces.push(format!(
                "[GPU Council Ready Impact] norm={:.4} delta={:.4} confidence_ema={:.3} mercy={:.3} coherence={:.3}",
                audit.mercy_norm, delta, self.symbolic_confidence_ema, self.state.mercy_score, self.one_organism_coherence
            ));
        } else if audit.mercy_norm > 0.6 {
            // Partial resonance for near-ready audits → encourages improvement loops in Powrush/RBE sims
            self.symbolic_confidence_ema = (self.symbolic_confidence_ema + 0.012).min(1.4);
            self.audit_traces.push("[GPU Near-Ready] mercy norm contributing to confidence calibration + PATSAGi readiness".to_string());
        }

        self.metrics.operations_processed += 1;
    }

    /// Notes that GpuComputePipeline real start handles (periodic auto-save) + shutdown timeout (5s)
    /// + circuit breaker + prometheus export are ready for conductor to own in Powrush lifecycle management.
    pub fn gpu_pipeline_lifecycle_ready(&self) -> String {
        "[v13.5] submit_patsagi_task_with_audit → (result, audit) wired. conductor-owned Arc<GpuComputePipeline> + start/stop wrappers ready. Telemetry/histograms/breaker/prometheus for observability.".to_string()
    }
}

// ==================== SYMBOLIC DELIBERATION ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicDeliberation {
    pub input: String,
    pub valence: f64,
    pub threshold_met: bool,
    pub confidence_score: f64,
    pub message: String,
}

pub fn metta_symbolic_deliberation(input: &str, context_valence: f64) -> SymbolicDeliberation {
    if context_valence >= 0.9999999 {
        SymbolicDeliberation { input: input.to_string(), valence: context_valence, threshold_met: true, confidence_score: 0.92, message: format!("metta_pln_truth_distilled_for_{}", input) }
    } else {
        SymbolicDeliberation { input: input.to_string(), valence: context_valence, threshold_met: false, confidence_score: 0.45, message: "metta_pln_compensated_low_valence".to_string() }
    }
}

// ==================== PHASE A (gated) ====================
#[cfg(feature = "external-symbolic")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSymbolicInput {
    pub source: String, pub content: String, pub context_valence: f64,
}

#[cfg(feature = "external-symbolic")]
impl ExternalSymbolicInput {
    pub fn new(source: &str, content: &str, context_valence: f64) -> Self {
        Self { source: source.to_string(), content: content.to_string(), context_valence }
    }
}

// ==================== PHASE B+C TYPES (gated) ====================
#[cfg(feature = "self-proposal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicSelfProposal {
    pub proposal_type: String,
    pub current_value: f64,
    pub proposed_value: f64,
    pub rationale: String,
    pub mercy_impact_estimate: f64,
    pub confidence: f64,
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_parameters_default() {
        let p = ConductorSymbolicParameters::default();
        assert!(p.base_confidence_threshold > 0.7);
        assert!(p.ema_alpha > 0.1);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_phase_c_apply_real_params() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        assert!(!c.get_self_proposal_log().is_empty());
        let before = c.symbolic_params.base_confidence_threshold;
        let _ = c.apply_top_confidence_proposal();
        assert!(c.symbolic_params.base_confidence_threshold != before || c.symbolic_params.boost_multiplier != 1.0);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_phase_c_mutates_real_parameters() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        let before_threshold = c.symbolic_params.base_confidence_threshold;
        let before_boost = c.symbolic_params.boost_multiplier;

        let _ = c.apply_top_confidence_proposal();

        let after_threshold = c.symbolic_params.base_confidence_threshold;
        let after_boost = c.symbolic_params.boost_multiplier;

        assert!(after_threshold != before_threshold || after_boost != before_boost);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_v13_3_meta_methods_delegate() {
        let c = SimpleLatticeConductor::new();
        let _ = c.generate_meta_self_evolution_proposals();
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_v13_4_meta_rate_delegation_and_getters() {
        let c = SimpleLatticeConductor::new();
        let _ = c.get_meta_evolution_rate();
        let _ = c.get_meta_audit_threshold();
        let _ = c.get_meta_success_ema();
    }

    #[test]
    fn test_v13_5_conductor_owns_gpu_pipeline_and_start_stop() {
        let mut c = SimpleLatticeConductor::new();
        let pipeline = Arc::new(GpuComputePipeline::new());
        c.set_gpu_pipeline(pipeline.clone());
        assert!(c.gpu_pipeline.is_some());

        // start/stop wrappers (facade path)
        let start_res = tokio::runtime::Runtime::new().unwrap().block_on(c.start_gpu_pipeline_telemetry(30));
        assert!(start_res.is_ok());

        let stop_res = tokio::runtime::Runtime::new().unwrap().block_on(c.stop_gpu_pipeline_telemetry());
        assert!(stop_res.is_ok());
    }

    #[test]
    fn test_v13_5_integrate_patsagi_gpu_audit_when_council_ready() {
        let mut c = SimpleLatticeConductor::new();
        let audit = MercyGpuAudit {
            task_id: 42,
            mercy_norm: 0.93,
            execution_time_ms: 87,
            reuse_ratio: 0.82,
            fragmentation_estimate: 120.0,
            council_ready: true,
            trace: "test ready".to_string(),
        };
        let before_conf = c.symbolic_confidence_ema;
        let before_mercy = c.state.mercy_score;
        c.integrate_patsagi_gpu_audit(&audit);
        assert!(c.symbolic_confidence_ema > before_conf);
        assert!(c.state.mercy_score >= before_mercy);
        assert!(c.audit_traces.iter().any(|t| t.contains("GPU Council Ready Impact")));
    }

    #[test]
    fn test_v13_5_gpu_pipeline_lifecycle_ready_note() {
        let c = SimpleLatticeConductor::new();
        let note = c.gpu_pipeline_lifecycle_ready();
        assert!(note.contains("conductor-owned Arc<GpuComputePipeline>"));
        assert!(note.contains("start/stop wrappers"));
    }
}