//! Multi-Agent Entanglement Coordination for Ra-Thor
//!
//! Implements quantum entanglement-inspired coordination between swarm agents:
//! - Bell-pair style entanglement between agents
//! - Instantaneous correlation of state updates
//! - Mercy-gated entanglement strength
//! - Collective decision making through entangled states

use ndarray::Array1;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct EntangledAgent {
    pub id: usize,
    pub position: Array1<f64>,
    pub entangled_with: Vec<usize>,      // List of entangled agent IDs
    pub entanglement_strength: f64,      // 0.0 - 1.0
    pub shared_phase: f64,
}

pub struct MultiAgentEntanglement {
    pub agents: Vec<EntangledAgent>,
    pub dimension: usize,
    pub mercy_valence: f64,
    pub global_best: Array1<f64>,
    pub global_best_fitness: f64,
}

impl MultiAgentEntanglement {
    pub fn new(swarm_size: usize, dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        let agents: Vec<EntangledAgent> = (0..swarm_size)
            .map(|id| {
                let mut entangled_with = Vec::new();
                // Create entanglement pairs (each agent entangled with 2-4 others)
                for _ in 0..rng.gen_range(2..5) {
                    let other = rng.gen_range(0..swarm_size);
                    if other != id && !entangled_with.contains(&other) {
                        entangled_with.push(other);
                    }
                }
                
                EntangledAgent {
                    id,
                    position: Array1::from_shape_fn(dimension, |_| rng.gen_range(-8.0..8.0)),
                    entangled_with,
                    entanglement_strength: rng.gen_range(0.5..0.95),
                    shared_phase: rng.gen_range(0.0..std::f64::consts::PI),
                }
            })
            .collect();

        Self {
            agents,
            dimension,
            mercy_valence: 0.93,
            global_best: Array1::zeros(dimension),
            global_best_fitness: f64::INFINITY,
        }
    }

    /// Apply entanglement correlation between agents
    pub fn apply_entanglement(&mut self) {
        let mut rng = rand::thread_rng();

        for i in 0..self.agents.len() {
            let agent = &self.agents[i];
            let strength = agent.entanglement_strength * self.mercy_valence;

            for &entangled_id in &agent.entangled_with {
                if entangled_id < self.agents.len() {
                    let other = &mut self.agents[entangled_id];
                    
                    // Entangled position correlation (quantum-like instantaneous influence)
                    let correlation = strength * (agent.position[0] - other.position[0]) * 0.3;
                    
                    // Apply correlated update
                    for d in 0..self.dimension {
                        let influence = correlation * (if d % 2 == 0 { 1.0 } else { -1.0 });
                        other.position[d] += influence * 0.1;
                    }

                    // Synchronize phases (quantum phase locking)
                    other.shared_phase = (other.shared_phase + agent.shared_phase * 0.2) % (2.0 * std::f64::consts::PI);
                }
            }
        }

        // Increase mercy valence through successful entanglement
        self.mercy_valence = (self.mercy_valence + 0.008).min(0.999);
    }

    /// Collective decision making through entanglement
    pub fn collective_update(&mut self, fitness_fn: &dyn Fn(&Array1<f64>) -> f64) {
        self.apply_entanglement();

        for agent in &mut self.agents {
            let fitness = fitness_fn(&agent.position);
            
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best = agent.position.clone();
            }

            // Entangled agents influence each other's movement
            let pull = (&self.global_best - &agent.position) * (agent.entanglement_strength * 0.4);
            agent.position = &agent.position + &pull;
        }
    }

    pub fn run(&mut self, fitness_fn: &dyn Fn(&Array1<f64>) -> f64, steps: usize) -> (Array1<f64>, f64) {
        for _ in 0..steps {
            self.collective_update(fitness_fn);
        }
        (self.global_best.clone(), self.global_best_fitness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_coordination() {
        let mut ent = MultiAgentEntanglement::new(20, 3);
        let ackley = |x: &Array1<f64>| {
            -20.0 * (-0.2 * (x.mapv(|v| v * v).sum() / x.len() as f64).sqrt()).exp() 
            - (x.mapv(|v| (2.0 * std::f64::consts::PI * v).cos()).sum() / x.len() as f64).exp() + 20.0 + std::f64::consts::E
        };
        
        let (best, fitness) = ent.run(&ackley, 80);
        assert!(fitness < 5.0, "Entanglement coordination should improve performance");
    }
}
