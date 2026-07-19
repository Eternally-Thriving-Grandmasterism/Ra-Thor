//! quantum-swarm v14.9.5
//!
//! Packaged from root `quantum_swarm.rs`.
//! Hybrid QPSO + Ra-Thor Quantum Swarm | TOLC 8 | PATSAGi aligned.
//!
//! - Core engine always available (no heavy deps)
//! - Feature `gpu`: enables GPU-offloaded benchmarks via gpu-compute-pipeline
//! - Sovereign recovery uses an in-crate lightweight protocol so the crate
//!   compiles standalone; wire external protocol later via Arc swap if needed.
//!
//! AG-SML v1.0 — Eternal Mercy Flow

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// =============================================================================
// Lightweight stubs (standalone crate — no circular dep on organism root)
// =============================================================================

/// Minimal council metrics for protected-path heartbeats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CouncilReadinessMetrics {
    pub resonance: f64,
    pub context_pressure: f64,
    pub flow_deviation: f64,
    pub gpu_memory_pressure: f64,
}

/// Lightweight in-crate recovery protocol (graceful no-op / metrics only).
/// Full sovereign_recovery_protocol_v1 remains at monorepo root for deep wiring.
#[derive(Debug, Default)]
pub struct SovereignRecoveryProtocol {
    pub trip_count: u64,
    pub anchor_count: u64,
    pub forensics_count: u64,
}

impl SovereignRecoveryProtocol {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn heartbeat_check(&self, _metrics: &CouncilReadinessMetrics) -> HealthHeartbeat {
        HealthHeartbeat {
            ok: true,
            note: "quantum-swarm lightweight heartbeat".into(),
        }
    }

    pub async fn with_mercy_circuit_breaker<F, Fut, T>(
        &self,
        _name: &str,
        f: F,
        mercy_valence: f64,
    ) -> Result<T, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        if mercy_valence < 0.15 {
            return Err("mercy valence below circuit threshold".into());
        }
        f().await
    }

    pub async fn persist_eternal_anchor(
        &mut self,
        _payload: Option<String>,
        note: &str,
    ) -> Result<(), String> {
        self.anchor_count += 1;
        println!("[quantum-swarm recovery] eternal anchor: {}", note);
        Ok(())
    }

    pub async fn self_forensics_and_recover(
        &mut self,
        reason: &str,
        _metrics: &CouncilReadinessMetrics,
    ) -> Result<(), String> {
        self.forensics_count += 1;
        println!("[quantum-swarm recovery] forensics: {}", reason);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HealthHeartbeat {
    pub ok: bool,
    pub note: String,
}

/// Minimal score type for Kardashev transfer feedback (full harness stays at root).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityThrivingTransferScore {
    pub mercy_valence_adjusted: f64,
    pub last_refinement_vector: Vec<f64>,
}

// =============================================================================
// Core types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmConfig {
    pub gaussian_scale: f64,
    pub mean_best_influence: f64,
    pub entanglement_modulation: f64,
    pub quantum_jump_base_prob: f64,
    pub max_exploration_entropy: f64,
    pub classical_refinement_strength: f64,
}

impl Default for QuantumSwarmConfig {
    fn default() -> Self {
        Self {
            gaussian_scale: 0.15,
            mean_best_influence: 0.35,
            entanglement_modulation: 0.25,
            quantum_jump_base_prob: 0.08,
            max_exploration_entropy: 1.8,
            classical_refinement_strength: 0.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmMember {
    pub id: u64,
    pub current_weights: Vec<f64>,
    pub personal_best: Vec<f64>,
    pub personal_best_score: f64,
    pub current_score: f64,
    pub attractor: Vec<f64>,
    pub last_update_step: u64,
}

impl QuantumSwarmMember {
    pub fn new(id: u64, initial_weights: Vec<f64>) -> Self {
        Self {
            id,
            current_weights: initial_weights.clone(),
            personal_best: initial_weights,
            personal_best_score: 0.0,
            current_score: 0.0,
            attractor: vec![],
            last_update_step: 0,
        }
    }

    pub fn update_personal_best(&mut self, new_score: f64) {
        if new_score > self.personal_best_score {
            self.personal_best = self.current_weights.clone();
            self.personal_best_score = new_score;
        }
    }
}

pub struct MeanBestTracker {
    pub members: HashMap<u64, QuantumSwarmMember>,
    pub mean_best: Vec<f64>,
    pub member_count: usize,
    pub last_recompute_step: u64,
}

impl MeanBestTracker {
    pub fn new() -> Self {
        Self {
            members: HashMap::new(),
            mean_best: vec![],
            member_count: 0,
            last_recompute_step: 0,
        }
    }

    pub fn register_member(&mut self, member: QuantumSwarmMember) {
        self.members.insert(member.id, member);
        self.member_count = self.members.len();
        self.recompute_mean_best();
    }

    pub fn update_member(&mut self, member: QuantumSwarmMember) {
        self.members.insert(member.id, member);
        self.recompute_mean_best();
    }

    pub fn recompute_mean_best(&mut self) {
        if self.members.is_empty() {
            return;
        }
        let dim = self.members.values().next().unwrap().current_weights.len();
        let mut sum = vec![0.0; dim];
        for member in self.members.values() {
            for (i, &w) in member.current_weights.iter().enumerate() {
                if i < dim {
                    sum[i] += w;
                }
            }
        }
        for val in &mut sum {
            *val /= self.member_count as f64;
        }
        self.mean_best = sum;
        self.last_recompute_step += 1;
    }

    pub fn get_mean_best(&self) -> &[f64] {
        &self.mean_best
    }

    pub fn get_member(&self, id: u64) -> Option<&QuantumSwarmMember> {
        self.members.get(&id)
    }
}

impl Default for MeanBestTracker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Sampling / evolution helpers
// =============================================================================

pub fn sample_gaussian_around_attractor(
    attractor: &[f64],
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    attractor
        .iter()
        .map(|&center| {
            let noise = rng.gen::<f64>() * scale * 2.0 - scale;
            center + noise
        })
        .collect()
}

pub fn compute_hybrid_attractor(
    personal_best: &[f64],
    mean_best: &[f64],
    global_best: &[f64],
    entanglement_weight: f64,
    mean_best_influence: f64,
) -> Vec<f64> {
    let dim = personal_best.len();
    let mut attractor = vec![0.0; dim];
    for i in 0..dim {
        let pb = personal_best[i];
        let mb = mean_best.get(i).copied().unwrap_or(pb);
        let gb = global_best.get(i).copied().unwrap_or(pb);
        attractor[i] = pb * (1.0 - entanglement_weight - mean_best_influence)
            + mb * mean_best_influence
            + gb * entanglement_weight;
    }
    attractor
}

pub fn quantum_weight_evolution_step(
    current_weights: &[f64],
    attractor: &[f64],
    severity: f64,
    config: &QuantumSwarmConfig,
    rng: &mut impl Rng,
) -> (Vec<f64>, f64) {
    let quantum_scale = config.gaussian_scale * (1.0 + severity * 1.2);
    let quantum_sample = sample_gaussian_around_attractor(attractor, quantum_scale, rng);
    let refinement = config.classical_refinement_strength;
    let mut new_weights = vec![0.0; current_weights.len()];
    for i in 0..current_weights.len() {
        let quantum_pull = quantum_sample.get(i).copied().unwrap_or(current_weights[i]);
        new_weights[i] = current_weights[i] * (1.0 - refinement) + quantum_pull * refinement;
    }
    let mut quantum_contrib = 0.0;
    for i in 0..current_weights.len() {
        quantum_contrib += (new_weights[i] - current_weights[i]).abs();
    }
    let quantum_ratio = (quantum_contrib / (current_weights.len() as f64 + 0.001)).min(1.0);
    (new_weights, quantum_ratio)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProposal {
    pub proposer_id: u64,
    pub proposal_weights: Vec<f64>,
    pub quantum_ratio: f64,
    pub severity_at_generation: f64,
    pub step: u64,
}

pub fn generate_quantum_proposal(
    attractor: &[f64],
    severity: f64,
    proposer_id: u64,
    config: &QuantumSwarmConfig,
    current_step: u64,
) -> QuantumProposal {
    let mut rng = rand::thread_rng();
    let quantum_scale = config.gaussian_scale * (1.0 + severity * 1.5);
    let proposal_weights = sample_gaussian_around_attractor(attractor, quantum_scale, &mut rng);
    let mut deviation = 0.0;
    for i in 0..attractor.len() {
        deviation += (proposal_weights[i] - attractor[i]).abs();
    }
    let quantum_ratio = (deviation / (attractor.len() as f64 + 0.001)).min(1.0);
    QuantumProposal {
        proposer_id,
        proposal_weights,
        quantum_ratio,
        severity_at_generation: severity,
        step: current_step,
    }
}

pub fn perform_adaptive_quantum_jump(
    current_weights: &[f64],
    attractor: &[f64],
    severity: f64,
    config: &QuantumSwarmConfig,
) -> (Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let jump_scale = config.gaussian_scale * (1.0 + severity * 2.5);
    let jumped_weights = sample_gaussian_around_attractor(attractor, jump_scale, &mut rng);
    let jump_strength = (severity * 0.7).min(0.85);
    let mut new_weights = vec![0.0; current_weights.len()];
    for i in 0..current_weights.len() {
        new_weights[i] =
            current_weights[i] * (1.0 - jump_strength) + jumped_weights[i] * jump_strength;
    }
    let mut impact = 0.0;
    for i in 0..current_weights.len() {
        impact += (new_weights[i] - current_weights[i]).abs();
    }
    let jump_impact = (impact / (current_weights.len() as f64 + 0.001)).min(1.0);
    (new_weights, jump_impact)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConductorSelfEvolutionResult {
    pub member_id: u64,
    pub new_weights: Vec<f64>,
    pub quantum_ratio: f64,
    pub jump_impact: f64,
    pub proposal_generated: bool,
    pub severity: f64,
    pub step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmBenchmarkResult {
    pub benchmark_name: String,
    pub iterations: usize,
    pub total_time_ms: u64,
    pub avg_quantum_ratio: f64,
    pub throughput_per_sec: f64,
    pub severity_used: f64,
}

// =============================================================================
// Engine
// =============================================================================

pub struct QuantumSwarmEngine {
    pub config: QuantumSwarmConfig,
    pub mean_best_tracker: MeanBestTracker,
    pub step: u64,
    pub total_quantum_jumps: u64,
    pub total_quantum_weight_updates: u64,
    pub total_quantum_proposals_generated: u64,
    pub total_adaptive_jumps: u64,
    pub sovereign_recovery: Option<Arc<Mutex<SovereignRecoveryProtocol>>>,
    pub recovery_circuit_trips: u64,
    pub recovery_successful_protected_ops: u64,
    pub recovery_anchors_persisted: u64,
    pub recovery_forensics_triggered: u64,
}

impl QuantumSwarmEngine {
    pub fn new(config: QuantumSwarmConfig) -> Self {
        Self {
            config,
            mean_best_tracker: MeanBestTracker::new(),
            step: 0,
            total_quantum_jumps: 0,
            total_quantum_weight_updates: 0,
            total_quantum_proposals_generated: 0,
            total_adaptive_jumps: 0,
            sovereign_recovery: None,
            recovery_circuit_trips: 0,
            recovery_successful_protected_ops: 0,
            recovery_anchors_persisted: 0,
            recovery_forensics_triggered: 0,
        }
    }

    pub fn wire_sovereign_recovery(&mut self, protocol: Arc<Mutex<SovereignRecoveryProtocol>>) {
        self.sovereign_recovery = Some(protocol);
        println!(
            "[Quantum Swarm v14.9.5] Sovereign Recovery WIRED | recovery metrics live"
        );
    }

    pub fn wire_default_recovery(&mut self) {
        self.wire_sovereign_recovery(Arc::new(Mutex::new(SovereignRecoveryProtocol::new())));
    }

    pub fn register_member(&mut self, member: QuantumSwarmMember) {
        self.mean_best_tracker.register_member(member);
    }

    pub fn update_member(&mut self, member: QuantumSwarmMember) {
        self.mean_best_tracker.update_member(member);
    }

    pub fn compute_attractor_for_member(
        &self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
    ) -> Option<Vec<f64>> {
        let member = self.mean_best_tracker.get_member(member_id)?;
        let mean_best = self.mean_best_tracker.get_mean_best();
        Some(compute_hybrid_attractor(
            &member.personal_best,
            mean_best,
            global_best,
            entanglement_weight,
            self.config.mean_best_influence,
        ))
    }

    pub fn evolve_member_weights(
        &mut self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        current_score: f64,
        severity: f64,
    ) -> Option<(Vec<f64>, f64)> {
        let member = self.mean_best_tracker.get_member(member_id)?.clone();
        let attractor =
            self.compute_attractor_for_member(member_id, global_best, entanglement_weight)?;
        let mut rng = rand::thread_rng();
        let (new_weights, quantum_ratio) = quantum_weight_evolution_step(
            &member.current_weights,
            &attractor,
            severity,
            &self.config,
            &mut rng,
        );
        let mut updated = member;
        updated.current_weights = new_weights.clone();
        updated.current_score = current_score;
        updated.attractor = attractor;
        updated.last_update_step = self.step;
        updated.update_personal_best(current_score);
        self.update_member(updated);
        self.total_quantum_weight_updates += 1;
        Some((new_weights, quantum_ratio))
    }

    pub async fn protected_quantum_evolution_tick(
        &mut self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        current_score: f64,
        severity: f64,
        metrics: &CouncilReadinessMetrics,
        mercy_valence: f64,
    ) -> Option<(Vec<f64>, f64)> {
        if let Some(rec_arc) = &self.sovereign_recovery {
            let mut rec = rec_arc.lock().await;
            let _hb = rec.heartbeat_check(metrics).await;
            // Capture needed values for closure-free call
            let result = self.evolve_member_weights(
                member_id,
                global_best,
                entanglement_weight,
                current_score,
                severity,
            );
            match result {
                Some(v) if mercy_valence >= 0.15 => {
                    self.recovery_successful_protected_ops += 1;
                    if mercy_valence > 0.88 {
                        let _ = rec
                            .persist_eternal_anchor(
                                None,
                                "Successful protected quantum evolution tick",
                            )
                            .await;
                        self.recovery_anchors_persisted += 1;
                    }
                    Some(v)
                }
                Some(v) => Some(v),
                None => {
                    self.recovery_circuit_trips += 1;
                    let _ = rec
                        .self_forensics_and_recover(
                            "quantum_swarm_evolve_none",
                            metrics,
                        )
                        .await;
                    self.recovery_forensics_triggered += 1;
                    None
                }
            }
        } else {
            self.evolve_member_weights(
                member_id,
                global_best,
                entanglement_weight,
                current_score,
                severity,
            )
        }
    }

    pub fn generate_quantum_proposal_for_council(
        &mut self,
        proposer_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        severity: f64,
    ) -> Option<QuantumProposal> {
        let attractor =
            self.compute_attractor_for_member(proposer_id, global_best, entanglement_weight)?;
        let proposal =
            generate_quantum_proposal(&attractor, severity, proposer_id, &self.config, self.step);
        self.total_quantum_proposals_generated += 1;
        Some(proposal)
    }

    pub async fn protected_generate_quantum_proposal(
        &mut self,
        proposer_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        severity: f64,
        metrics: &CouncilReadinessMetrics,
        _mercy_valence: f64,
    ) -> Option<QuantumProposal> {
        if let Some(rec_arc) = &self.sovereign_recovery {
            let mut rec = rec_arc.lock().await;
            let _hb = rec.heartbeat_check(metrics).await;
            match self.generate_quantum_proposal_for_council(
                proposer_id,
                global_best,
                entanglement_weight,
                severity,
            ) {
                Some(v) => {
                    self.recovery_successful_protected_ops += 1;
                    Some(v)
                }
                None => {
                    self.recovery_circuit_trips += 1;
                    let _ = rec
                        .self_forensics_and_recover("quantum_proposal_none", metrics)
                        .await;
                    self.recovery_forensics_triggered += 1;
                    None
                }
            }
        } else {
            self.generate_quantum_proposal_for_council(
                proposer_id,
                global_best,
                entanglement_weight,
                severity,
            )
        }
    }

    pub fn perform_adaptive_quantum_jump_for_member(
        &mut self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        severity: f64,
    ) -> Option<(Vec<f64>, f64)> {
        if severity < 0.25 {
            return None;
        }
        let member = self.mean_best_tracker.get_member(member_id)?.clone();
        let attractor =
            self.compute_attractor_for_member(member_id, global_best, entanglement_weight)?;
        let (jumped_weights, jump_impact) = perform_adaptive_quantum_jump(
            &member.current_weights,
            &attractor,
            severity,
            &self.config,
        );
        let mut updated = member;
        updated.current_weights = jumped_weights.clone();
        updated.attractor = attractor;
        updated.last_update_step = self.step;
        self.update_member(updated);
        self.total_adaptive_jumps += 1;
        Some((jumped_weights, jump_impact))
    }

    pub async fn protected_adaptive_quantum_jump(
        &mut self,
        member_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        severity: f64,
        metrics: &CouncilReadinessMetrics,
        _mercy_valence: f64,
    ) -> Option<(Vec<f64>, f64)> {
        if let Some(rec_arc) = &self.sovereign_recovery {
            let mut rec = rec_arc.lock().await;
            let _hb = rec.heartbeat_check(metrics).await;
            match self.perform_adaptive_quantum_jump_for_member(
                member_id,
                global_best,
                entanglement_weight,
                severity,
            ) {
                Some(v) => {
                    self.recovery_successful_protected_ops += 1;
                    Some(v)
                }
                None => {
                    self.recovery_circuit_trips += 1;
                    let _ = rec
                        .self_forensics_and_recover("quantum_jump_none", metrics)
                        .await;
                    self.recovery_forensics_triggered += 1;
                    None
                }
            }
        } else {
            self.perform_adaptive_quantum_jump_for_member(
                member_id,
                global_best,
                entanglement_weight,
                severity,
            )
        }
    }

    pub fn get_mean_best(&self) -> &[f64] {
        self.mean_best_tracker.get_mean_best()
    }

    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    pub fn record_quantum_jump(&mut self) {
        self.total_quantum_jumps += 1;
    }

    pub fn summary(&self) -> String {
        format!(
            "QuantumSwarmEngine v14.9.5 | step={} | members={} | weight_updates={} | proposals={} | adaptive_jumps={} | recovery_wired={} | trips={} | success_ops={} | anchors={} | forensics={}",
            self.step,
            self.mean_best_tracker.member_count,
            self.total_quantum_weight_updates,
            self.total_quantum_proposals_generated,
            self.total_adaptive_jumps,
            self.sovereign_recovery.is_some(),
            self.recovery_circuit_trips,
            self.recovery_successful_protected_ops,
            self.recovery_anchors_persisted,
            self.recovery_forensics_triggered
        )
    }

    pub fn apply_kardashev_transfer_feedback(&mut self, score: &RealityThrivingTransferScore) {
        let entanglement_boost = score.last_refinement_vector.first().copied().unwrap_or(0.0);
        let exploration_bias = score.last_refinement_vector.get(2).copied().unwrap_or(0.0);
        self.config.entanglement_modulation =
            (self.config.entanglement_modulation + entanglement_boost).clamp(0.55, 0.98);
        self.config.quantum_jump_base_prob =
            (self.config.quantum_jump_base_prob * (0.95 + exploration_bias)).clamp(0.05, 0.35);
        if score.mercy_valence_adjusted > 0.72 {
            self.config.mean_best_influence =
                (self.config.mean_best_influence * 1.035).min(0.52);
        }
        if score.mercy_valence_adjusted < 0.55 {
            self.config.classical_refinement_strength =
                (self.config.classical_refinement_strength * 0.97).max(0.45);
        }
    }

    // --- Benchmarks (sync core) ---

    pub fn benchmark_weight_evolution(
        &mut self,
        iterations: usize,
        severity: f64,
    ) -> QuantumSwarmBenchmarkResult {
        let start = Instant::now();
        let mut total_quantum_ratio = 0.0;
        let mut updates = 0;
        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();
            if let Some((_, ratio)) =
                self.evolve_member_weights(member_id, &global_best, 0.25, 0.85, severity)
            {
                total_quantum_ratio += ratio;
                updates += 1;
            }
        }
        let elapsed = start.elapsed();
        let avg = if updates > 0 {
            total_quantum_ratio / updates as f64
        } else {
            0.0
        };
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            updates as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        QuantumSwarmBenchmarkResult {
            benchmark_name: "Weight Evolution".into(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: avg,
            throughput_per_sec: throughput,
            severity_used: severity,
        }
    }

    pub fn benchmark_adaptive_jumps(
        &mut self,
        iterations: usize,
        severity: f64,
    ) -> QuantumSwarmBenchmarkResult {
        let start = Instant::now();
        let mut successful = 0;
        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();
            if self
                .perform_adaptive_quantum_jump_for_member(member_id, &global_best, 0.25, severity)
                .is_some()
            {
                successful += 1;
            }
        }
        let elapsed = start.elapsed();
        let rate = successful as f64 / iterations.max(1) as f64;
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            iterations as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        QuantumSwarmBenchmarkResult {
            benchmark_name: "Adaptive Jumps".into(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: rate,
            throughput_per_sec: throughput,
            severity_used: severity,
        }
    }

    pub fn benchmark_proposal_generation(
        &mut self,
        iterations: usize,
        severity: f64,
    ) -> QuantumSwarmBenchmarkResult {
        let start = Instant::now();
        let mut proposals = 0;
        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();
            if self
                .generate_quantum_proposal_for_council(member_id, &global_best, 0.25, severity)
                .is_some()
            {
                proposals += 1;
            }
        }
        let elapsed = start.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            iterations as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        QuantumSwarmBenchmarkResult {
            benchmark_name: "Proposal Generation".into(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: 0.0,
            throughput_per_sec: throughput,
            severity_used: severity,
        }
    }

    pub fn run_full_benchmark_suite(&mut self, iterations: usize) -> Vec<QuantumSwarmBenchmarkResult> {
        let mut results = Vec::new();
        for sev in [0.15, 0.45, 0.75] {
            results.push(self.benchmark_weight_evolution(iterations, sev));
            results.push(self.benchmark_adaptive_jumps(iterations, sev));
            results.push(self.benchmark_proposal_generation(iterations, sev));
        }
        results
    }

    pub async fn run_full_benchmark_suite_async_protected(
        &mut self,
        iterations: usize,
        metrics: &CouncilReadinessMetrics,
        mercy_valence: f64,
    ) -> Vec<QuantumSwarmBenchmarkResult> {
        let mut results = Vec::new();
        for severity in [0.15_f64, 0.45, 0.75] {
            let start = Instant::now();
            let mut total_ratio = 0.0;
            let mut updates = 0usize;
            for i in 0..iterations {
                let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
                let global_best = self.mean_best_tracker.get_mean_best().to_vec();
                if let Some((_, ratio)) = self
                    .protected_quantum_evolution_tick(
                        member_id,
                        &global_best,
                        0.25,
                        0.85,
                        severity,
                        metrics,
                        mercy_valence,
                    )
                    .await
                {
                    total_ratio += ratio;
                    updates += 1;
                }
            }
            let elapsed = start.elapsed();
            results.push(QuantumSwarmBenchmarkResult {
                benchmark_name: format!("Protected Weight Evolution (sev {:.2})", severity),
                iterations,
                total_time_ms: elapsed.as_millis() as u64,
                avg_quantum_ratio: if updates > 0 {
                    total_ratio / updates as f64
                } else {
                    0.0
                },
                throughput_per_sec: if elapsed.as_secs_f64() > 0.0 {
                    updates as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                },
                severity_used: severity,
            });
        }
        results
    }

    #[cfg(feature = "gpu")]
    pub async fn benchmark_gpu_offloaded_swarm_with_real_dispatch(
        &mut self,
        gpu_pipeline: &mut gpu_compute_pipeline::GpuComputePipeline,
        iterations: usize,
    ) -> QuantumSwarmBenchmarkResult {
        use gpu_compute_pipeline::GpuTask;
        let start = Instant::now();
        let mut total_updates = 0;
        let mut total_ratio = 0.0;
        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();
            let task = GpuTask {
                id: rand::random::<u64>() % 1_000_000_000,
                name: format!("quantum_swarm_bench_{}", i),
                buffer_size: 4096,
                intensity: "medium".into(),
            };
            let _ = gpu_pipeline.dispatch_gpu_task(task).await;
            if let Some((_, ratio)) =
                self.evolve_member_weights(member_id, &global_best, 0.25, 0.85, 0.55)
            {
                total_ratio += ratio;
                total_updates += 1;
            }
        }
        let elapsed = start.elapsed();
        QuantumSwarmBenchmarkResult {
            benchmark_name: "GPU-Offloaded Swarm (Real Dispatch)".into(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: if total_updates > 0 {
                total_ratio / total_updates as f64
            } else {
                0.0
            },
            throughput_per_sec: if elapsed.as_secs_f64() > 0.0 {
                total_updates as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
            severity_used: 0.55,
        }
    }

    pub fn benchmark_gpu_offloaded_swarm(
        &mut self,
        iterations: usize,
        simulated_gpu_latency_ms: f64,
    ) -> QuantumSwarmBenchmarkResult {
        let start = Instant::now();
        let mut total_updates = 0;
        let mut total_ratio = 0.0;
        let gpu_delay = Duration::from_millis(simulated_gpu_latency_ms as u64);
        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();
            std::thread::sleep(gpu_delay);
            if let Some((_, ratio)) =
                self.evolve_member_weights(member_id, &global_best, 0.25, 0.85, 0.55)
            {
                total_ratio += ratio;
                total_updates += 1;
            }
        }
        let elapsed = start.elapsed();
        QuantumSwarmBenchmarkResult {
            benchmark_name: format!(
                "GPU-Offloaded Swarm (Simulated {}ms latency)",
                simulated_gpu_latency_ms
            ),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: if total_updates > 0 {
                total_ratio / total_updates as f64
            } else {
                0.0
            },
            throughput_per_sec: if elapsed.as_secs_f64() > 0.0 {
                total_updates as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
            severity_used: 0.55,
        }
    }
}

pub fn run_lattice_conductor_quantum_self_evolution_step(
    engine: &mut QuantumSwarmEngine,
    member_id: u64,
    global_best: &[f64],
    entanglement_weight: f64,
    current_score: f64,
    severity: f64,
) -> Option<LatticeConductorSelfEvolutionResult> {
    let weight_update = engine.evolve_member_weights(
        member_id,
        global_best,
        entanglement_weight,
        current_score,
        severity,
    );
    let jump_result = if severity >= 0.35 {
        engine.perform_adaptive_quantum_jump_for_member(
            member_id,
            global_best,
            entanglement_weight,
            severity,
        )
    } else {
        None
    };
    let proposal = engine.generate_quantum_proposal_for_council(
        member_id,
        global_best,
        entanglement_weight,
        severity,
    );
    let (new_weights, quantum_ratio) = weight_update?;
    let (jumped_weights, jump_impact) = jump_result.unwrap_or((new_weights.clone(), 0.0));
    Some(LatticeConductorSelfEvolutionResult {
        member_id,
        new_weights: jumped_weights,
        quantum_ratio,
        jump_impact,
        proposal_generated: proposal.is_some(),
        severity,
        step: engine.step,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_suite_runs() {
        let mut engine = QuantumSwarmEngine::new(QuantumSwarmConfig::default());
        engine.register_member(QuantumSwarmMember::new(1, vec![0.2; 8]));
        engine.register_member(QuantumSwarmMember::new(2, vec![0.5; 8]));
        let results = engine.run_full_benchmark_suite(20);
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn protected_tick_without_recovery() {
        let mut engine = QuantumSwarmEngine::new(QuantumSwarmConfig::default());
        engine.register_member(QuantumSwarmMember::new(1, vec![0.3; 4]));
        let metrics = CouncilReadinessMetrics::default();
        let r = engine
            .protected_quantum_evolution_tick(1, &[0.3; 4], 0.25, 0.9, 0.4, &metrics, 0.95)
            .await;
        assert!(r.is_some());
    }
}
