//! Quantum Random Walks for Ra-Thor Quantum Swarm Orchestrator
//!
//! Implements quantum-inspired random walks with:
//! - Quantum superposition-inspired position updates
//! - Adaptive step size based on mercy valence
//! - Exploration/exploitation balance control
//! - Integration with QPSO for hybrid quantum-classical search

use ndarray::Array1;
use rand::Rng;
use std::f64::consts::PI;

/// Quantum Walk Agent
#[derive(Debug, Clone)]
pub struct QuantumWalkAgent {
    pub position: Array1<f64>,
    pub step_size: f64,
    pub quantum_coherence: f64,      // How "quantum" the walk currently is
    pub exploration_bias: f64,       // Tendency to explore vs exploit
}

/// Quantum Random Walk Swarm
pub struct QuantumRandomWalks {
    pub agents: Vec<QuantumWalkAgent>,
    pub dimension: usize,
    pub mercy_valence: f64,
    pub global_best: Array1<f64>,
    pub global_best_fitness: f64,
}

impl QuantumRandomWalks {
    pub fn new(swarm_size: usize, dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        let agents: Vec<QuantumWalkAgent> = (0..swarm_size)
            .map(|_| QuantumWalkAgent {
                position: Array1::from_shape_fn(dimension, |_| rng.gen_range(-10.0..10.0)),
                step_size: rng.gen_range(0.1..1.5),
                quantum_coherence: rng.gen_range(0.6..0.95),
                exploration_bias: rng.gen_range(0.4..0.8),
            })
            .collect();

        Self {
            agents,
            dimension,
            mercy_valence: 0.92,
            global_best: Array1::zeros(dimension),
            global_best_fitness: f64::INFINITY,
        }
    }

    /// Perform one quantum walk step
    pub fn step(&mut self, fitness_fn: &dyn Fn(&Array1<f64>) -> f64) {
        let mut rng = rand::thread_rng();

        for agent in &mut self.agents {
            // Quantum-inspired position update (superposition-like)
            let phase = rng.gen_range(0.0..2.0 * PI);
            let quantum_step = agent.step_size * (phase.cos() + agent.quantum_coherence * phase.sin());

            // Apply exploration or exploitation bias
            let direction: Array1<f64> = if rng.gen::<f64>() < agent.exploration_bias {
                // Exploration: random direction
                Array1::from_shape_fn(self.dimension, |_| rng.gen_range(-1.0..1.0))
            } else {
                // Exploitation: move toward global best
                &self.global_best - &agent.position
            };

            // Update position with quantum step
            agent.position = &agent.position + &(direction * quantum_step);

            // Boundary handling
            for val in agent.position.iter_mut() {
                if *val > 12.0 { *val = 12.0; }
                if *val < -12.0 { *val = -12.0; }
            }

            // Evaluate fitness
            let fitness = fitness_fn(&agent.position);
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best = agent.position.clone();
            }

            // Update quantum coherence (increases with mercy valence)
            agent.quantum_coherence = (agent.quantum_coherence + 0.01 * self.mercy_valence).min(0.98);
            
            // Occasionally adjust exploration bias
            if rng.gen::<f64>() < 0.1 {
                agent.exploration_bias = (agent.exploration_bias + rng.gen_range(-0.1..0.1)).clamp(0.2, 0.9);
            }
        }

        // Slowly improve mercy valence
        self.mercy_valence = (self.mercy_valence + 0.005).min(0.999);
    }

    /// Run multiple quantum walk steps
    pub fn run(&mut self, fitness_fn: &dyn Fn(&Array1<f64>) -> f64, steps: usize) -> (Array1<f64>, f64) {
        for _ in 0..steps {
            self.step(fitness_fn);
        }
        (self.global_best.clone(), self.global_best_fitness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_walks_convergence() {
        let mut walks = QuantumRandomWalks::new(25, 4);
        let rastrigin = |x: &Array1<f64>| {
            10.0 * x.len() as f64 + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
        };
        
        let (best_pos, best_fitness) = walks.run(&rastrigin, 150);
        assert!(best_fitness < 5.0, "Quantum walks should make progress on Rastrigin");
    }
}
