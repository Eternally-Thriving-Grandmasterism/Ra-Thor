// quantum_swarm.rs
// Ra-Thor v14.59 — Quantum Swarm Optimization (Phase 2: QPSO-Style Weight Evolution — A)
// Hybrid QPSO + Ra-Thor Quantum Swarm
// Lattice Conductor v13.1 | ONE Organism | PATSAGi Councils
//
// Phase 2 Complete: QPSO-style weight evolution integrated into the Quantum Swarm Engine.
// Foundation from Phase 1 (E + D) is now actively used for self-evolving weights.
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

// === Mean Best Position (Direction D) ===

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

// === Phase 2: QPSO-Style Weight Evolution (Direction A) ===

/// Performs a hybrid quantum + classical weight evolution step.
/// This is the core QPSO-inspired update for Lattice Conductor self-evolution.
pub fn quantum_weight_evolution_step(
    current_weights: &mut [f64],
    attractor: &[f64],
    severity: f64,
    config: &QuantumSwarmConfig,
    rng: &mut impl rand::Rng,
) -> (Vec<f64>, f64) {
    // 1. Quantum sampling around the hybrid attractor
    let quantum_scale = config.gaussian_scale * (1.0 + severity * 1.2);
    let quantum_sample = sample_gaussian_around_attractor(attractor, quantum_scale, rng);

    // 2. Classical refinement (momentum-style pull toward quantum sample)
    let refinement = config.classical_refinement_strength;
    let mut new_weights = vec![0.0; current_weights.len()];

    for i in 0..current_weights.len() {
        let quantum_pull = quantum_sample[i];
        let classical = current_weights[i] * (1.0 - refinement) + quantum_pull * refinement;
        new_weights[i] = classical;
    }

    // 3. Compute quantum contribution ratio for telemetry
    let mut quantum_contrib = 0.0;
    for i in 0..current_weights.len() {
        let diff = (new_weights[i] - current_weights[i]).abs();
        quantum_contrib += diff;
    }
    let quantum_ratio = (quantum_contrib / (current_weights.len() as f64 + 0.001)).min(1.0);

    (new_weights, quantum_ratio)
}

// === Quantum Swarm Engine (Phase 2 Enhanced) ===

pub struct QuantumSwarmEngine {
    pub config: QuantumSwarmConfig,
    pub mean_best_tracker: MeanBestTracker,
    pub step: u64,
    pub total_quantum_jumps: u64,
    pub total_quantum_weight_updates: u64,
}

impl QuantumSwarmEngine {
    pub fn new(config: QuantumSwarmConfig) -> Self {
        Self {
            config,
            mean_best_tracker: MeanBestTracker::new(),
            step: 0,
            total_quantum_jumps: 0,
            total_quantum_weight_updates: 0,
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

    /// Phase 2: Execute full QPSO-style weight evolution step for a member
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

        // Update member state
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
            "QuantumSwarmEngine v14.59 | step={} | members={} | quantum_updates={} | quantum_jumps={}",
            self.step,
            self.mean_best_tracker.member_count,
            self.total_quantum_weight_updates,
            self.total_quantum_jumps
        )
    }
}

// === Integration Hook for Lattice Conductor Self-Evolution ===
// Call `evolve_member_weights(...)` from Lattice Conductor self-evolution loops.
// Pass current severity (from plateau detection) to modulate quantum exploration.
// The returned quantum_ratio can be logged for telemetry and mercy gating.

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
    fn test_quantum_weight_evolution_step() {
        let mut weights = vec![0.5, 0.5];
        let attractor = vec![0.7, 0.3];
        let config = QuantumSwarmConfig::default();
        let mut rng = thread_rng();

        let (new_w, ratio) = quantum_weight_evolution_step(&mut weights, &attractor, 0.4, &config, &mut rng);

        assert!(new_w.len() == 2);
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_hybrid_attractor() {
        let pb = vec![0.8, 0.1];
        let mb = vec![0.5, 0.5];
        let gb = vec![0.9, 0.0];

        let attractor = compute_hybrid_attractor(&pb, &mb, &gb, 0.3, 0.35);
        assert!(attractor.len() == 2);
    }
}
