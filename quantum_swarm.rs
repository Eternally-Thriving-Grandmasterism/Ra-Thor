// quantum_swarm.rs
// Ra-Thor v14.65 — Quantum Swarm Engine Benchmark Suite (Extended)
// Hybrid QPSO + Ra-Thor Quantum Swarm
// Lattice Conductor v13.1 | ONE Organism | PATSAGi Councils
//
// Extended with scaling benchmarks + mean best recompute cost.
// Benchmark telemetry wired into Lattice Conductor self-evolution.
//
// Perfect order of operations. Thunder locked in.
//
// AG-SML v1.0 License

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

// === Core Quantum Swarm Types ===

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

// === Mean Best Position ===

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
                sum[i] += w;
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

// === Quantum Sampling ===

pub fn sample_gaussian_around_attractor(
    attractor: &[f64],
    scale: f64,
    rng: &mut impl rand::Rng,
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
        let mb = if !mean_best.is_empty() { mean_best[i] } else { pb };
        let gb = if !global_best.is_empty() { global_best[i] } else { pb };

        let blended = pb * (1.0 - entanglement_weight - mean_best_influence)
            + mb * mean_best_influence
            + gb * entanglement_weight;

        attractor[i] = blended;
    }

    attractor
}

// === Phase 2: QPSO-Style Weight Evolution ===

pub fn quantum_weight_evolution_step(
    current_weights: &mut [f64],
    attractor: &[f64],
    severity: f64,
    config: &QuantumSwarmConfig,
    rng: &mut impl rand::Rng,
) -> (Vec<f64>, f64) {
    let quantum_scale = config.gaussian_scale * (1.0 + severity * 1.2);
    let quantum_sample = sample_gaussian_around_attractor(attractor, quantum_scale, rng);

    let refinement = config.classical_refinement_strength;
    let mut new_weights = vec![0.0; current_weights.len()];

    for i in 0..current_weights.len() {
        let quantum_pull = quantum_sample[i];
        let classical = current_weights[i] * (1.0 - refinement) + quantum_pull * refinement;
        new_weights[i] = classical;
    }

    let mut quantum_contrib = 0.0;
    for i in 0..current_weights.len() {
        let diff = (new_weights[i] - current_weights[i]).abs();
        quantum_contrib += diff;
    }
    let quantum_ratio = (quantum_contrib / (current_weights.len() as f64 + 0.001)).min(1.0);

    (new_weights, quantum_ratio)
}

// === Phase 3: Quantum Proposal / Vote Generation ===

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
    use rand::thread_rng;
    let mut rng = thread_rng();

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

// === Phase 4: Adaptive Quantum Jumps ===

pub fn should_perform_adaptive_quantum_jump(severity: f64, base_prob: f64) -> bool {
    use rand::thread_rng;
    let mut rng = thread_rng();

    let jump_prob = base_prob + (severity * 0.45);
    let clamped = jump_prob.min(0.92);
    rng.gen::<f64>() < clamped
}

pub fn perform_adaptive_quantum_jump(
    current_weights: &mut [f64],
    attractor: &[f64],
    severity: f64,
    config: &QuantumSwarmConfig,
) -> (Vec<f64>, f64) {
    use rand::thread_rng;
    let mut rng = thread_rng();

    let jump_scale = config.gaussian_scale * (1.0 + severity * 2.5);
    let jumped_weights = sample_gaussian_around_attractor(attractor, jump_scale, &mut rng);

    let jump_strength = (severity * 0.7).min(0.85);
    let mut new_weights = vec![0.0; current_weights.len()];

    for i in 0..current_weights.len() {
        new_weights[i] = current_weights[i] * (1.0 - jump_strength) + jumped_weights[i] * jump_strength;
    }

    let mut impact = 0.0;
    for i in 0..current_weights.len() {
        impact += (new_weights[i] - current_weights[i]).abs();
    }
    let jump_impact = (impact / (current_weights.len() as f64 + 0.001)).min(1.0);

    (new_weights, jump_impact)
}

// === Phase 5: Lattice Conductor Wiring ===

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

// === Quantum Swarm Engine (with Extended Benchmarks) ===

pub struct QuantumSwarmEngine {
    pub config: QuantumSwarmConfig,
    pub mean_best_tracker: MeanBestTracker,
    pub step: u64,
    pub total_quantum_jumps: u64,
    pub total_quantum_weight_updates: u64,
    pub total_quantum_proposals_generated: u64,
    pub total_adaptive_jumps: u64,
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
        }
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

        let attractor = compute_hybrid_attractor(
            &member.personal_best,
            mean_best,
            global_best,
            entanglement_weight,
            self.config.mean_best_influence,
        );

        Some(attractor)
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

        let attractor = self.compute_attractor_for_member(member_id, global_best, entanglement_weight)?;

        use rand::thread_rng;
        let mut rng = thread_rng();

        let (new_weights, quantum_ratio) = quantum_weight_evolution_step(
            &member.current_weights,
            &attractor,
            severity,
            &self.config,
            &mut rng,
        );

        let mut updated_member = member.clone();
        updated_member.current_weights = new_weights.clone();
        updated_member.current_score = current_score;
        updated_member.attractor = attractor;
        updated_member.last_update_step = self.step;
        updated_member.update_personal_best(current_score);

        self.update_member(updated_member);
        self.total_quantum_weight_updates += 1;

        Some((new_weights, quantum_ratio))
    }

    pub fn generate_quantum_proposal_for_council(
        &mut self,
        proposer_id: u64,
        global_best: &[f64],
        entanglement_weight: f64,
        severity: f64,
    ) -> Option<QuantumProposal> {
        let attractor = self.compute_attractor_for_member(proposer_id, global_best, entanglement_weight)?;

        let proposal = generate_quantum_proposal(
            &attractor,
            severity,
            proposer_id,
            &self.config,
            self.step,
        );

        self.total_quantum_proposals_generated += 1;
        Some(proposal)
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
        let attractor = self.compute_attractor_for_member(member_id, global_best, entanglement_weight)?;

        let (jumped_weights, jump_impact) = perform_adaptive_quantum_jump(
            &member.current_weights,
            &attractor,
            severity,
            &self.config,
        );

        let mut updated_member = member.clone();
        updated_member.current_weights = jumped_weights.clone();
        updated_member.attractor = attractor;
        updated_member.last_update_step = self.step;

        self.update_member(updated_member);
        self.total_adaptive_jumps += 1;

        Some((jumped_weights, jump_impact))
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
            "QuantumSwarmEngine v14.65 | step={} | members={} | weight_updates={} | proposals={} | adaptive_jumps={}",
            self.step,
            self.mean_best_tracker.member_count,
            self.total_quantum_weight_updates,
            self.total_quantum_proposals_generated,
            self.total_adaptive_jumps
        )
    }

    // === Extended Benchmark Suite ===

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

            if let Some((_, ratio)) = self.evolve_member_weights(
                member_id,
                &global_best,
                0.25,
                0.85,
                severity,
            ) {
                total_quantum_ratio += ratio;
                updates += 1;
            }
        }

        let elapsed = start.elapsed();
        let avg_quantum_ratio = if updates > 0 { total_quantum_ratio / updates as f64 } else { 0.0 };
        let throughput = if elapsed.as_secs_f64() > 0.0 { updates as f64 / elapsed.as_secs_f64() } else { 0.0 };

        QuantumSwarmBenchmarkResult {
            benchmark_name: "Weight Evolution".to_string(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio,
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
        let mut successful_jumps = 0;

        for i in 0..iterations {
            let member_id = ((i % self.mean_best_tracker.member_count.max(1)) + 1) as u64;
            let global_best = self.mean_best_tracker.get_mean_best().to_vec();

            if self.perform_adaptive_quantum_jump_for_member(member_id, &global_best, 0.25, severity).is_some() {
                successful_jumps += 1;
            }
        }

        let elapsed = start.elapsed();
        let jump_success_rate = successful_jumps as f64 / iterations as f64;
        let throughput = if elapsed.as_secs_f64() > 0.0 { iterations as f64 / elapsed.as_secs_f64() } else { 0.0 };

        QuantumSwarmBenchmarkResult {
            benchmark_name: "Adaptive Jumps".to_string(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: jump_success_rate,
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

            if self.generate_quantum_proposal_for_council(member_id, &global_best, 0.25, severity).is_some() {
                proposals += 1;
            }
        }

        let elapsed = start.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 { iterations as f64 / elapsed.as_secs_f64() } else { 0.0 };

        QuantumSwarmBenchmarkResult {
            benchmark_name: "Proposal Generation".to_string(),
            iterations,
            total_time_ms: elapsed.as_millis() as u64,
            avg_quantum_ratio: 0.0,
            throughput_per_sec: throughput,
            severity_used: severity,
        }
    }

    // NEW: Scaling benchmark (performance vs number of members)
    pub fn benchmark_scaling_with_members(&mut self, max_members: usize, iterations_per_member: usize) -> Vec<QuantumSwarmBenchmarkResult> {
        let mut results = Vec::new();

        for member_count in (1..=max_members).step_by(2).chain(std::iter::once(max_members)) {
            // Reset engine for clean scaling test
            let config = self.config.clone();
            let mut fresh_engine = QuantumSwarmEngine::new(config);

            for i in 1..=member_count {
                fresh_engine.register_member(QuantumSwarmMember::new(i as u64, vec![0.1 * i as f64; 8]));
            }

            let start = Instant::now();
            let mut total_updates = 0;

            for _ in 0..iterations_per_member {
                for mid in 1..=member_count {
                    let global_best = fresh_engine.get_mean_best().to_vec();
                    if fresh_engine.evolve_member_weights(mid as u64, &global_best, 0.25, 0.85, 0.45).is_some() {
                        total_updates += 1;
                    }
                }
            }

            let elapsed = start.elapsed();
            let throughput = if elapsed.as_secs_f64() > 0.0 { total_updates as f64 / elapsed.as_secs_f64() } else { 0.0 };

            results.push(QuantumSwarmBenchmarkResult {
                benchmark_name: format!("Scaling ({} members)", member_count),
                iterations: iterations_per_member * member_count,
                total_time_ms: elapsed.as_millis() as u64,
                avg_quantum_ratio: 0.0,
                throughput_per_sec: throughput,
                severity_used: 0.45,
            });
        }

        results
    }

    // NEW: Mean Best recompute cost benchmark
    pub fn benchmark_mean_best_recompute_cost(&mut self, member_counts: Vec<usize>) -> Vec<QuantumSwarmBenchmarkResult> {
        let mut results = Vec::new();

        for &count in &member_counts {
            let config = self.config.clone();
            let mut fresh_engine = QuantumSwarmEngine::new(config);

            for i in 1..=count {
                fresh_engine.register_member(QuantumSwarmMember::new(i as u64, vec![0.05 * i as f64; 8]));
            }

            let start = Instant::now();
            for _ in 0..100 {
                fresh_engine.mean_best_tracker.recompute_mean_best();
            }
            let elapsed = start.elapsed();

            let avg_time_us = (elapsed.as_micros() as f64) / 100.0;

            results.push(QuantumSwarmBenchmarkResult {
                benchmark_name: format!("MeanBest Recompute ({} members)", count),
                iterations: 100,
                total_time_ms: elapsed.as_millis() as u64,
                avg_quantum_ratio: avg_time_us,
                throughput_per_sec: 0.0,
                severity_used: 0.0,
            });
        }

        results
    }

    pub fn run_full_benchmark_suite(&mut self, iterations: usize) -> Vec<QuantumSwarmBenchmarkResult> {
        println!("\n[Quantum Swarm Engine Benchmark Suite] Starting full benchmark ({} iterations per test)...", iterations);

        let mut results = Vec::new();

        results.push(self.benchmark_weight_evolution(iterations, 0.15));
        results.push(self.benchmark_adaptive_jumps(iterations, 0.15));
        results.push(self.benchmark_proposal_generation(iterations, 0.15));

        results.push(self.benchmark_weight_evolution(iterations, 0.45));
        results.push(self.benchmark_adaptive_jumps(iterations, 0.45));
        results.push(self.benchmark_proposal_generation(iterations, 0.45));

        results.push(self.benchmark_weight_evolution(iterations, 0.75));
        results.push(self.benchmark_adaptive_jumps(iterations, 0.75));
        results.push(self.benchmark_proposal_generation(iterations, 0.75));

        println!("\n=== Quantum Swarm Engine Benchmark Results ===");
        for r in &results {
            println!(
                "[{}] | {} iters | {:.2} ms | Throughput: {:.1}/s | QuantumRatio/JumpRate: {:.3} | Severity: {:.2}",
                r.benchmark_name, r.iterations, r.total_time_ms, r.throughput_per_sec, r.avg_quantum_ratio, r.severity_used
            );
        }

        results
    }
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

// === Tests ===
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_benchmark_suite() {
        let mut engine = QuantumSwarmEngine::new(QuantumSwarmConfig::default());
        engine.register_member(QuantumSwarmMember::new(1, vec![0.2; 8]));
        engine.register_member(QuantumSwarmMember::new(2, vec![0.5; 8]));

        let results = engine.run_full_benchmark_suite(300);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_scaling_benchmark() {
        let mut engine = QuantumSwarmEngine::new(QuantumSwarmConfig::default());
        let scaling = engine.benchmark_scaling_with_members(8, 50);
        assert!(scaling.len() >= 4);
    }
}
