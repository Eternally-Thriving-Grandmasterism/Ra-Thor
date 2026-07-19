//! Extended Organism Surface — v14.10.0
//!
//! Facades + optional live path binding for packaged workspace crates:
//! - GPU compute telemetry          (`gpu-live`)
//! - GitHub evolution connector     (`github-live` — offline-first + optional flush)
//! - Quantum Swarm engine           (`quantum-live` — full evolution cycle)
//! - Sovereign Recovery             (`recovery-live`)
//! - Kardashev / Reality Transfer   (`kardashev-live`)
//!
//! When a live feature is enabled the corresponding surface holds the real
//! crate engine and routes through its async APIs via a safe block_on helper.
//! Facades remain the zero-dep default when features are off.
//!
//! Cosmic Loop remains mandatory on every surface call.
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

use crate::CouncilArbitrationEngine;

// =============================================================================
// Live runtime helper (shared by all live paths)
// =============================================================================

#[cfg(any(
    feature = "recovery-live",
    feature = "kardashev-live",
    feature = "gpu-live",
    feature = "quantum-live",
    feature = "github-live"
))]
fn block_on_live<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(fut),
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("[ExtendedSurface] failed to build fallback runtime for live path");
            rt.block_on(fut)
        }
    }
}

// =============================================================================
// GPU Surface — live path when gpu-live
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
    pub live_path: bool,
}

#[derive(Debug)]
pub struct GpuSurface {
    dispatch_count: u64,
    total_dispatch_time_ms: u64,
    last_telemetry: Option<GpuDispatchTelemetry>,
    pool_efficiency: f64,
    memory_usage_bytes: usize,
    #[cfg(feature = "gpu-live")]
    pipeline: std::sync::Arc<tokio::sync::Mutex<gpu_compute_pipeline::GpuComputePipeline>>,
}

impl Clone for GpuSurface {
    fn clone(&self) -> Self {
        Self {
            dispatch_count: self.dispatch_count,
            total_dispatch_time_ms: self.total_dispatch_time_ms,
            last_telemetry: self.last_telemetry.clone(),
            pool_efficiency: self.pool_efficiency,
            memory_usage_bytes: self.memory_usage_bytes,
            #[cfg(feature = "gpu-live")]
            pipeline: self.pipeline.clone(),
        }
    }
}

impl GpuSurface {
    pub fn new() -> Self {
        Self {
            dispatch_count: 0,
            total_dispatch_time_ms: 0,
            last_telemetry: None,
            pool_efficiency: 0.93,
            memory_usage_bytes: 512 * 1024 * 1024,
            #[cfg(feature = "gpu-live")]
            pipeline: std::sync::Arc::new(tokio::sync::Mutex::new(
                gpu_compute_pipeline::GpuComputePipeline::new(),
            )),
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

        #[cfg(feature = "gpu-live")]
        {
            use gpu_compute_pipeline::GpuTask;
            let task = GpuTask {
                id: self.dispatch_count + 1,
                name: task_name.into(),
                buffer_size: elements.max(64),
                intensity: if elements > 8192 {
                    "high".into()
                } else {
                    "medium".into()
                },
            };
            let result = block_on_live(async {
                let mut pipe = self.pipeline.lock().await;
                if real_gpu {
                    pipe.mark_real_gpu(true);
                }
                pipe.dispatch_gpu_task(task).await
            });

            self.dispatch_count += 1;
            let measured = result.execution_time_ms.max(1);
            self.total_dispatch_time_ms += measured;

            let workgroups = ((elements + 63) / 64) as u32;
            let tel = GpuDispatchTelemetry {
                task_id: self.dispatch_count,
                task_name: task_name.into(),
                real_gpu: result.real_gpu,
                dispatch_time_ms: measured,
                readback_available: result.readback_data.is_some(),
                elements_processed: elements,
                workgroups_dispatched: workgroups,
            };
            self.last_telemetry = Some(tel.clone());

            if measured > 80 {
                println!(
                    "[GpuSurface LIVE] anomaly: {}ms on {} — Debugger handoff recommended",
                    measured, task_name
                );
            }
            return tel;
        }

        #[cfg(not(feature = "gpu-live"))]
        {
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
    }

    pub fn status(&self) -> GpuSurfaceStatus {
        GpuSurfaceStatus {
            dispatch_count: self.dispatch_count,
            total_dispatch_time_ms: self.total_dispatch_time_ms,
            last_telemetry: self.last_telemetry.clone(),
            pool_efficiency: self.pool_efficiency,
            memory_usage_bytes: self.memory_usage_bytes,
            live_path: cfg!(feature = "gpu-live"),
        }
    }
}

impl Default for GpuSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GitHub Surface — offline-first + optional live flush when github-live
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
    pub live_path: bool,
    pub connector_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlushResult {
    pub title: String,
    pub success: bool,
    pub html_url: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct GitHubSurface {
    intended_prs: Vec<EvolutionPrIntent>,
    offline_mode: bool,
    #[cfg(feature = "github-live")]
    connector: Option<std::sync::Arc<github_connector::GitHubConnector>>,
}

impl Clone for GitHubSurface {
    fn clone(&self) -> Self {
        Self {
            intended_prs: self.intended_prs.clone(),
            offline_mode: self.offline_mode,
            #[cfg(feature = "github-live")]
            connector: self.connector.clone(),
        }
    }
}

impl GitHubSurface {
    pub fn new() -> Self {
        #[cfg(feature = "github-live")]
        {
            let connector = match github_connector::GitHubConnector::from_env(
                "Eternally-Thriving-Grandmasterism",
                "Ra-Thor",
            ) {
                Ok(c) => {
                    println!(
                        "[GitHubSurface LIVE] Connector ready | owner={} repo={} | rate_limit={}",
                        c.owner(),
                        c.repo(),
                        c.get_rate_limit_remaining()
                    );
                    Some(std::sync::Arc::new(c))
                }
                Err(e) => {
                    println!(
                        "[GitHubSurface] No live token ({}), staying offline-first. Queue only.",
                        e.message
                    );
                    None
                }
            };
            let offline = connector.is_none();
            return Self {
                intended_prs: Vec::new(),
                offline_mode: offline,
                connector,
            };
        }

        #[cfg(not(feature = "github-live"))]
        {
            Self {
                intended_prs: Vec::new(),
                offline_mode: true,
            }
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

    pub fn flush_to_github(
        &mut self,
        arbitration: &CouncilArbitrationEngine,
    ) -> Vec<FlushResult> {
        arbitration.enforce_cosmic_loop_activation();
        arbitration.before_council_arbitration();

        let intents = self.drain_intents();
        if intents.is_empty() {
            return Vec::new();
        }

        #[cfg(feature = "github-live")]
        {
            if let Some(conn) = &self.connector {
                let mut results = Vec::with_capacity(intents.len());
                for intent in intents {
                    let res = block_on_live(conn.create_role_optimized_evolution_pr(
                        &intent.role,
                        &intent.target_module,
                        &intent.description,
                        intent.expected_benefit,
                        intent.mercy_alignment,
                    ));
                    match res {
                        Ok(pr) => {
                            println!(
                                "[GitHubSurface LIVE] PR #{} opened | {}",
                                pr.number, pr.html_url
                            );
                            results.push(FlushResult {
                                title: intent.title,
                                success: true,
                                html_url: Some(pr.html_url),
                                error: None,
                            });
                        }
                        Err(e) => {
                            eprintln!(
                                "[GitHubSurface LIVE] Mercy Circuit: failed to open PR for '{}': {}",
                                intent.title, e.message
                            );
                            results.push(FlushResult {
                                title: intent.title,
                                success: false,
                                html_url: None,
                                error: Some(e.message),
                            });
                        }
                    }
                }
                return results;
            }
        }

        intents
            .into_iter()
            .map(|i| FlushResult {
                title: i.title,
                success: false,
                html_url: None,
                error: Some("offline_mode or no live connector".into()),
            })
            .collect()
    }

    pub fn status(&self) -> GitHubSurfaceStatus {
        GitHubSurfaceStatus {
            intended_prs: self.intended_prs.len(),
            last_intent: self.intended_prs.last().cloned(),
            offline_mode: self.offline_mode,
            live_path: cfg!(feature = "github-live"),
            #[cfg(feature = "github-live")]
            connector_ready: self.connector.is_some(),
            #[cfg(not(feature = "github-live"))]
            connector_ready: false,
        }
    }
}

impl Default for GitHubSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Quantum Swarm Surface — full evolution cycle when quantum-live
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
    pub live_path: bool,
}

/// Rich result of a full quantum evolution cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEvolutionResult {
    pub step: u64,
    pub member_id: u64,
    pub quantum_ratio: f64,
    pub jump_impact: f64,
    pub proposal_generated: bool,
    pub severity: f64,
    pub weight_update_ok: bool,
}

#[derive(Debug)]
pub struct QuantumSwarmSurface {
    config: QuantumSwarmConfig,
    step: u64,
    member_count: usize,
    total_weight_updates: u64,
    total_adaptive_jumps: u64,
    total_proposals: u64,
    #[cfg(feature = "quantum-live")]
    engine: std::sync::Arc<tokio::sync::Mutex<quantum_swarm::QuantumSwarmEngine>>,
}

impl Clone for QuantumSwarmSurface {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            step: self.step,
            member_count: self.member_count,
            total_weight_updates: self.total_weight_updates,
            total_adaptive_jumps: self.total_adaptive_jumps,
            total_proposals: self.total_proposals,
            #[cfg(feature = "quantum-live")]
            engine: self.engine.clone(),
        }
    }
}

impl QuantumSwarmSurface {
    pub fn new() -> Self {
        #[cfg(feature = "quantum-live")]
        {
            use quantum_swarm::{QuantumSwarmConfig as QConfig, QuantumSwarmEngine, QuantumSwarmMember};
            let mut eng = QuantumSwarmEngine::new(QConfig::default());
            for id in 1..=4 {
                eng.register_member(QuantumSwarmMember::new(id, vec![0.25 + id as f64 * 0.05; 8]));
            }
            eng.wire_default_recovery();
            return Self {
                config: QuantumSwarmConfig::default(),
                step: 0,
                member_count: 4,
                total_weight_updates: 0,
                total_adaptive_jumps: 0,
                total_proposals: 0,
                engine: std::sync::Arc::new(tokio::sync::Mutex::new(eng)),
            };
        }

        #[cfg(not(feature = "quantum-live"))]
        {
            Self {
                config: QuantumSwarmConfig::default(),
                step: 0,
                member_count: 0,
                total_weight_updates: 0,
                total_adaptive_jumps: 0,
                total_proposals: 0,
            }
        }
    }

    pub fn register_members(&mut self, count: usize) {
        self.member_count = count;
        #[cfg(feature = "quantum-live")]
        {
            use quantum_swarm::QuantumSwarmMember;
            let mut eng = block_on_live(self.engine.lock());
            for id in 1..=count as u64 {
                if eng.mean_best_tracker.get_member(id).is_none() {
                    eng.register_member(QuantumSwarmMember::new(
                        id,
                        vec![0.2 + (id as f64) * 0.03; 8],
                    ));
                }
            }
        }
    }

    /// Full quantum evolution cycle:
    /// 1. Protected weight evolution (always)
    /// 2. Adaptive quantum jump when severity >= 0.35
    /// 3. Proposal generation when severity >= 0.15
    ///
    /// Under `quantum-live` this routes through real `QuantumSwarmEngine` APIs.
    pub fn evolve_full_cycle(
        &mut self,
        severity: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> QuantumEvolutionResult {
        arbitration.enforce_cosmic_loop_activation();
        self.step += 1;
        let severity = severity.clamp(0.0, 1.0);
        let member_id = ((self.step % self.member_count.max(1) as u64) + 1).max(1);

        #[cfg(feature = "quantum-live")]
        {
            use quantum_swarm::CouncilReadinessMetrics;
            let metrics = CouncilReadinessMetrics {
                resonance: 0.94,
                context_pressure: (severity * 0.4).min(1.0),
                flow_deviation: (severity * 0.25).min(0.5),
                gpu_memory_pressure: 0.3,
            };

            let (ratio, jump_impact, proposal_ok, weight_ok) = block_on_live(async {
                let mut eng = self.engine.lock().await;
                eng.increment_step();
                let global_best = eng.get_mean_best().to_vec();

                // 1. Protected weight evolution
                let weight_res = eng
                    .protected_quantum_evolution_tick(
                        member_id,
                        &global_best,
                        0.25,
                        0.88,
                        severity,
                        &metrics,
                        0.95,
                    )
                    .await;
                let (ratio, weight_ok) = match weight_res {
                    Some((_, r)) => (r, true),
                    None => ((eng.config.gaussian_scale * (1.0 + severity)).min(1.0), false),
                };

                // 2. Adaptive jump
                let mut jump_impact = 0.0;
                if severity >= 0.35 {
                    if let Some((_, impact)) = eng
                        .protected_adaptive_quantum_jump(
                            member_id,
                            &global_best,
                            0.25,
                            severity,
                            &metrics,
                            0.95,
                        )
                        .await
                    {
                        jump_impact = impact;
                    }
                }

                // 3. Proposal generation
                let mut proposal_ok = false;
                if severity >= 0.15 {
                    if eng
                        .protected_generate_quantum_proposal(
                            member_id,
                            &global_best,
                            0.25,
                            severity,
                            &metrics,
                            0.95,
                        )
                        .await
                        .is_some()
                    {
                        proposal_ok = true;
                    }
                }

                // Sync local counters from engine when possible
                // (engine tracks its own totals; we keep surface-level mirrors)
                (ratio, jump_impact, proposal_ok, weight_ok)
            });

            if weight_ok {
                self.total_weight_updates += 1;
            }
            if jump_impact > 0.0 {
                self.total_adaptive_jumps += 1;
            }
            if proposal_ok {
                self.total_proposals += 1;
            }

            // Keep local config loosely in sync with defaults / severity pressure
            self.config.gaussian_scale =
                (self.config.gaussian_scale * 0.97 + 0.03 * (0.12 + severity * 0.08)).clamp(0.08, 0.35);

            return QuantumEvolutionResult {
                step: self.step,
                member_id,
                quantum_ratio: ratio,
                jump_impact,
                proposal_generated: proposal_ok,
                severity,
                weight_update_ok: weight_ok,
            };
        }

        #[cfg(not(feature = "quantum-live"))]
        {
            self.total_weight_updates += 1;
            let mut jump_impact = 0.0;
            if severity >= 0.35 {
                self.total_adaptive_jumps += 1;
                jump_impact = (severity * 0.55).min(0.85);
            }
            let mut proposal_ok = false;
            if severity >= 0.15 {
                self.total_proposals += 1;
                proposal_ok = true;
            }
            let ratio = (self.config.gaussian_scale * (1.0 + severity)).min(1.0);
            QuantumEvolutionResult {
                step: self.step,
                member_id,
                quantum_ratio: ratio,
                jump_impact,
                proposal_generated: proposal_ok,
                severity,
                weight_update_ok: true,
            }
        }
    }

    /// Lightweight tick — returns quantum_ratio only (compatible with Cosmic Tick).
    /// Internally runs the full evolution cycle.
    pub fn evolution_tick(
        &mut self,
        severity: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> f64 {
        self.evolve_full_cycle(severity, arbitration).quantum_ratio
    }

    /// Apply Kardashev / Reality Thriving Transfer feedback into the swarm config.
    /// Under `quantum-live` this calls the real engine's `apply_kardashev_transfer_feedback`.
    pub fn apply_kardashev_feedback(
        &mut self,
        transfer: &TransferTickResult,
        arbitration: &CouncilArbitrationEngine,
    ) {
        arbitration.enforce_cosmic_loop_activation();

        #[cfg(feature = "quantum-live")]
        {
            use quantum_swarm::RealityThrivingTransferScore;
            // Map organism TransferTickResult → engine score shape
            let score = RealityThrivingTransferScore {
                mercy_valence_adjusted: transfer.ema_transfer,
                last_refinement_vector: vec![
                    (transfer.kardashev_delta * 40.0).clamp(-0.08, 0.08),
                    (0.5 - transfer.ethics_index) * 0.05,
                    (transfer.abundance_velocity - 0.9) * 0.04,
                ],
            };
            let mut eng = block_on_live(self.engine.lock());
            eng.apply_kardashev_transfer_feedback(&score);
            // Mirror a couple of knobs locally for status
            self.config.entanglement_modulation = eng.config.entanglement_modulation;
            self.config.quantum_jump_base_prob = eng.config.quantum_jump_base_prob;
            self.config.mean_best_influence = eng.config.mean_best_influence;
            println!(
                "[QuantumSwarmSurface LIVE] Kardashev feedback applied | entanglement={:.3} jump_prob={:.3}",
                self.config.entanglement_modulation, self.config.quantum_jump_base_prob
            );
            return;
        }

        #[cfg(not(feature = "quantum-live"))]
        {
            // Facade-side gentle modulation
            let boost = (transfer.kardashev_delta * 30.0).clamp(-0.05, 0.05);
            self.config.entanglement_modulation =
                (self.config.entanglement_modulation + boost).clamp(0.15, 0.55);
            if transfer.ema_transfer > 0.72 {
                self.config.mean_best_influence =
                    (self.config.mean_best_influence * 1.02).min(0.52);
            }
        }
    }

    pub fn status(&self) -> QuantumSwarmStatus {
        QuantumSwarmStatus {
            step: self.step,
            member_count: self.member_count,
            total_weight_updates: self.total_weight_updates,
            total_adaptive_jumps: self.total_adaptive_jumps,
            total_proposals: self.total_proposals,
            config: self.config.clone(),
            live_path: cfg!(feature = "quantum-live"),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "QuantumSwarmSurface v14.10.0 | step={} | members={} | updates={} | jumps={} | proposals={} | live={}",
            self.step,
            self.member_count,
            self.total_weight_updates,
            self.total_adaptive_jumps,
            self.total_proposals,
            cfg!(feature = "quantum-live")
        )
    }
}

impl Default for QuantumSwarmSurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Sovereign Recovery Surface — live path when recovery-live
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
    pub live_path: bool,
}

#[derive(Debug)]
pub struct SovereignRecoverySurface {
    heartbeat_count: u64,
    anchor_count: u64,
    recovery_events: u64,
    last_heartbeat: Option<RecoveryHeartbeat>,
    last_anchor: Option<RecoveryAnchor>,
    #[cfg(feature = "recovery-live")]
    protocol: std::sync::Arc<sovereign_recovery::SovereignRecoveryProtocol>,
}

impl Clone for SovereignRecoverySurface {
    fn clone(&self) -> Self {
        Self {
            heartbeat_count: self.heartbeat_count,
            anchor_count: self.anchor_count,
            recovery_events: self.recovery_events,
            last_heartbeat: self.last_heartbeat.clone(),
            last_anchor: self.last_anchor.clone(),
            #[cfg(feature = "recovery-live")]
            protocol: self.protocol.clone(),
        }
    }
}

impl SovereignRecoverySurface {
    pub fn new() -> Self {
        Self {
            heartbeat_count: 0,
            anchor_count: 0,
            recovery_events: 0,
            last_heartbeat: None,
            last_anchor: None,
            #[cfg(feature = "recovery-live")]
            protocol: std::sync::Arc::new(sovereign_recovery::SovereignRecoveryProtocol::new()),
        }
    }

    pub fn heartbeat(
        &mut self,
        mercy_norm: f64,
        gpu_confidence: f64,
        tick: u64,
        arbitration: &CouncilArbitrationEngine,
    ) -> RecoveryHeartbeat {
        arbitration.enforce_cosmic_loop_activation();
        self.heartbeat_count += 1;

        #[cfg(feature = "recovery-live")]
        {
            use sovereign_recovery::CouncilReadinessMetrics;
            let metrics = CouncilReadinessMetrics {
                mercy_norm,
                gpu_mercy_modulated_confidence: gpu_confidence,
                gpu_memory_usage_bytes: 256 * 1024 * 1024,
                council_ready: true,
                council_resonance: 0.94,
                last_gpu_readback_available: gpu_confidence > 0.75,
                last_updated_tick: tick,
                evolution_level: 1,
            };
            let live_hb = block_on_live(self.protocol.heartbeat_check(&metrics));
            let hb = RecoveryHeartbeat {
                context_pressure: live_hb.context_pressure,
                flow_deviation: live_hb.flow_state_deviation,
                connector_health: live_hb.connector_health,
                requires_recovery: live_hb.requires_recovery,
                tick,
            };
            if hb.requires_recovery {
                self.recovery_events += 1;
            }
            self.last_heartbeat = Some(hb.clone());
            return hb;
        }

        #[cfg(not(feature = "recovery-live"))]
        {
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
    }

    pub fn persist_anchor(
        &mut self,
        note: &str,
        tick: u64,
        arbitration: &CouncilArbitrationEngine,
    ) -> RecoveryAnchor {
        arbitration.enforce_cosmic_loop_activation();
        self.anchor_count += 1;

        #[cfg(feature = "recovery-live")]
        {
            let live_anchor =
                block_on_live(self.protocol.persist_eternal_anchor(None, note));
            let anchor = RecoveryAnchor {
                anchor_id: live_anchor.anchor_id,
                note: live_anchor.recovery_note,
                tick,
            };
            self.last_anchor = Some(anchor.clone());
            return anchor;
        }

        #[cfg(not(feature = "recovery-live"))]
        {
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
    }

    pub fn status(&self) -> SovereignRecoveryStatus {
        SovereignRecoveryStatus {
            heartbeat_count: self.heartbeat_count,
            anchor_count: self.anchor_count,
            last_heartbeat: self.last_heartbeat.clone(),
            last_anchor: self.last_anchor.clone(),
            recovery_events: self.recovery_events,
            live_path: cfg!(feature = "recovery-live"),
        }
    }
}

impl Default for SovereignRecoverySurface {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Kardashev / Reality Thriving Transfer Surface — live path when kardashev-live
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
    pub live_path: bool,
}

#[derive(Debug)]
pub struct KardashevFlywheelSurface {
    cycle_count: u64,
    cumulative_kardashev_delta: f64,
    velocity_ema: f64,
    last_transfer: Option<TransferTickResult>,
    #[cfg(feature = "kardashev-live")]
    calculator: std::sync::Arc<reality_thriving_transfer::RealityThrivingTransferCalculator>,
    #[cfg(feature = "kardashev-live")]
    council: std::sync::Arc<kardashev_orchestration::KardashevOrchestrationCouncil>,
}

impl Clone for KardashevFlywheelSurface {
    fn clone(&self) -> Self {
        Self {
            cycle_count: self.cycle_count,
            cumulative_kardashev_delta: self.cumulative_kardashev_delta,
            velocity_ema: self.velocity_ema,
            last_transfer: self.last_transfer.clone(),
            #[cfg(feature = "kardashev-live")]
            calculator: self.calculator.clone(),
            #[cfg(feature = "kardashev-live")]
            council: self.council.clone(),
        }
    }
}

impl KardashevFlywheelSurface {
    pub fn new() -> Self {
        Self {
            cycle_count: 0,
            cumulative_kardashev_delta: 0.0,
            velocity_ema: 0.42,
            last_transfer: None,
            #[cfg(feature = "kardashev-live")]
            calculator: std::sync::Arc::new(
                reality_thriving_transfer::RealityThrivingTransferCalculator::new(),
            ),
            #[cfg(feature = "kardashev-live")]
            council: std::sync::Arc::new(
                kardashev_orchestration::KardashevOrchestrationCouncil::new(),
            ),
        }
    }

    pub fn transfer_tick(
        &mut self,
        rbe_quality: f64,
        ethical_choice: f64,
        abundance_signal: f64,
        arbitration: &CouncilArbitrationEngine,
    ) -> TransferTickResult {
        arbitration.enforce_cosmic_loop_activation();
        self.cycle_count += 1;

        #[cfg(feature = "kardashev-live")]
        {
            use reality_thriving_transfer::PowrushTelemetry;

            let telemetry = PowrushTelemetry {
                gameplay_hours: 120.0 + self.cycle_count as f64 * 0.5,
                rbe_decision_quality_avg: rbe_quality.clamp(0.0, 1.0),
                peaceful_resolution_rate: ((rbe_quality + ethical_choice) / 2.0).clamp(0.0, 1.0),
                collaboration_events: 400 + (self.cycle_count * 3),
                ethical_choice_score: ethical_choice.clamp(0.0, 1.0),
                adaptation_events: 90 + self.cycle_count,
                abundance_velocity_signals: abundance_signal.max(0.0),
                innovation_contribution: (rbe_quality * 0.6 + ethical_choice * 0.4).clamp(0.0, 1.0),
            };

            let score = match block_on_live(self.calculator.compute_transfer_score(&telemetry)) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[KardashevFlywheelSurface LIVE] Mercy Gate: {}", e);
                    let result = TransferTickResult {
                        ema_transfer: 0.0,
                        kardashev_delta: 0.0,
                        abundance_velocity: self.velocity_ema,
                        ethics_index: 0.0,
                        mercy_audit_passed: false,
                        cycle: self.cycle_count,
                    };
                    self.last_transfer = Some(result.clone());
                    return result;
                }
            };

            let deliberation =
                block_on_live(self.council.deliberate_acceleration_cycle(&[score.clone()], None));

            self.cumulative_kardashev_delta = deliberation.cumulative_kardashev_delta;
            self.velocity_ema = deliberation.abundance_velocity_trend;

            let result = TransferTickResult {
                ema_transfer: score.ema_refined_transfer,
                kardashev_delta: score.kardashev_delta_contribution,
                abundance_velocity: deliberation.abundance_velocity_trend,
                ethics_index: score.ethics_collaboration_index,
                mercy_audit_passed: deliberation.mercy_audit_passed,
                cycle: self.cycle_count,
            };
            self.last_transfer = Some(result.clone());
            println!(
                "[KardashevFlywheelSurface LIVE] cycle={} Δ={:.5} velocity={:.3} ethics_pass={} inflection={}",
                self.cycle_count,
                result.kardashev_delta,
                result.abundance_velocity,
                result.mercy_audit_passed,
                deliberation.s_curve_projection.inflection_year
            );
            return result;
        }

        #[cfg(not(feature = "kardashev-live"))]
        {
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
            live_path: cfg!(feature = "kardashev-live"),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "KardashevFlywheel v14.10.0 | cycles={} | Δ={:.5} | velocity={:.3} | inflection={} | live={}",
            self.cycle_count,
            self.cumulative_kardashev_delta,
            self.velocity_ema,
            self.projected_inflection_year(),
            cfg!(feature = "kardashev-live")
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
            "ExtendedSurface v14.10.0 | {} | recovery_hb={} anchors={} live_rec={} | github_offline={} | {}",
            self.quantum_swarm.summary(),
            self.sovereign_recovery.heartbeat_count,
            self.sovereign_recovery.anchor_count,
            cfg!(feature = "recovery-live"),
            self.github.offline_mode,
            self.kardashev.summary()
        )
    }
}

impl Default for ExtendedOrganismSurface {
    fn default() -> Self {
        Self::new()
    }
}
