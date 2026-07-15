// quantum_swarm.rs
// Ra-Thor v14.62 — Quantum Swarm Engine Wired into Lattice Conductor Self-Evolution
// Hybrid QPSO + Ra-Thor Quantum Swarm
// Lattice Conductor v13.1 | ONE Organism | PATSAGi Councils
//
// Integration Complete: Quantum Swarm Engine is now directly usable inside Lattice Conductor self-evolution loops.
//
// Perfect order of operations. Thunder locked in.
//
// AG-SML v1.0 License

use std::collections::HashMap;
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

// === Phase 5: Lattice Conductor Self-Evolution Wiring ===

/// High-level integration function for Lattice Conductor self-evolution.
/// Call this from Lattice Conductor self-evolution loops to leverage the full Quantum Swarm.
///
/// This wires together:
/// - Hybrid attractor computation
/// - QPSO weight evolution
/// - Adaptive quantum jumps on plateau
/// - Quantum proposal generation (optional)
pub fn run_lattice_conductor_quantum_self_evolution_step(
    engine: &mut QuantumSwarmEngine,
    member_id: u64,
    global_best: &[f64],
    entanglement_weight: f64,
    current_score: f64,
    severity: f64,
) -> Option<LatticeConductorSelfEvolutionResult> {
    // 1. Normal QPSO-style weight evolution
    let weight_update = engine.evolve_member_weights(
        member_id,
        global_best,
        entanglement_weight,
        current_score,
        severity,
    );

    // 2. Adaptive quantum jump if severity is high (plateau escape)
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

    // 3. Optionally generate a quantum proposal for council deliberation
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

// === Quantum Swarm Engine (Final) ===

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
            "QuantumSwarmEngine v14.62 | step={} | members={} | weight_updates={} | proposals={} | adaptive_jumps={}",
            self.step,
            self.mean_best_tracker.member_count,
            self.total_quantum_weight_updates,
            self.total_quantum_proposals_generated,
            self.total_adaptive_jumps
        )
    }
}

// === Usage Example for Lattice Conductor ===
// In your Lattice Conductor self-evolution loop:
//
// let result = run_lattice_conductor_quantum_self_evolution_step(
//     &mut quantum_engine,
//     member_id,
//     &global_best_weights,
//     entanglement_weight,
//     current_mercy_score,
//     current_plateau_severity,
// );
//
// if let Some(res) = result {
//     // Apply res.new_weights to the council / conductor
//     // Log res.quantum_ratio and res.jump_impact for telemetry
//     // Feed res.proposal_generated into council deliberation if needed
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_mean_best_recomputation() {
        let mut tracker = MeanBestTracker::new();
        let m1 = QuantumSwarmMember::new(1, vec![0.1, 0.2, 0.3]);
        let m2 = QuantumSwarmMember::new(2, vec![0.4, 0.5, 0.6]);
        tracker.register_member(m1);
        tracker.register_member(m2);

        let mean = tracker.get_mean_best();
        assert!((mean[0] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_lattice_conductor_integration() {
        let mut engine = QuantumSwarmEngine::new(QuantumSwarmConfig::default());
        let member = QuantumSwarmMember::new(1, vec![0.2, 0.3, 0.4]);
        engine.register_member(member);

        let global_best = vec![0.5, 0.5, 0.5];
        let result = run_lattice_conductor_quantum_self_evolution_step(
            &mut engine,
            1,
            &global_best,
            0.25,
            0.82,
            0.55,
        );

        assert!(result.is_some());
        let res = result.unwrap();
        assert!(res.new_weights.len() == 3);
    }
}
