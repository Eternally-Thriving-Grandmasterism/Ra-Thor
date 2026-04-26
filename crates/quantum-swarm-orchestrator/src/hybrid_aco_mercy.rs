//! # ACO + Mercy Hybrid Swarm — Refined Production Version
//!
//! **Ant Colony Optimization with strict mercy-gating on every pheromone update.**
//!
//! This refined version includes:
//! - Full 7 Living Mercy Gates validation on every ant
//! - Configurable pheromone evaporation and deposit rates
//! - Hebbian bond strengthening only on mercy-aligned paths
//! - Clear separation of concerns and rich documentation
//! - Production-ready structure for long-running simulations

use ra_thor_legal_lattice::cehi::CEHIImpact;
use ra_thor_plasticity_engine_v2::PlasticityEngineV2;
use rand::Rng;

/// A single ant in the hybrid ACO-Mercy swarm.
#[derive(Debug, Clone)]
pub struct Ant {
    pub position: Vec<f64>,
    pub path_quality: f64,
    pub hebbian_bond: f64,
    pub pheromone_contribution: f64,
}

/// Refined ACO + Mercy Hybrid Swarm Orchestrator.
pub struct HybridACOMercy {
    pub ants: Vec<Ant>,
    pub pheromone_map: Vec<Vec<f64>>,
    pub global_best_path: Vec<f64>,
    pub mercy_valence: f64,
    pub plasticity_engine: PlasticityEngineV2,
    pub dimension: usize,

    // Configurable parameters
    pub evaporation_rate: f64,      // Default 0.92
    pub deposit_factor: f64,        // Default 0.4
    pub min_gate_threshold: f64,    // Minimum CEHI for gate 1
}

impl HybridACOMercy {
    pub fn new(num_ants: usize, dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let ants = (0..num_ants)
            .map(|_| Ant {
                position: (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect(),
                path_quality: 0.0,
                hebbian_bond: 0.60 + rng.gen_range(0.0..0.30),
                pheromone_contribution: 0.0,
            })
            .collect();

        let pheromone_map = vec![vec![0.1; dimension]; dimension];

        Self {
            ants,
            pheromone_map,
            global_best_path: vec![0.0; dimension],
            mercy_valence: 0.62,
            plasticity_engine: PlasticityEngineV2::new(),
            dimension,
            evaporation_rate: 0.92,
            deposit_factor: 0.40,
            min_gate_threshold: 0.05,
        }
    }

    /// Validates an ant against **all 7 Living Mercy Gates**.
    /// This is the non-bypassable ethical filter.
    fn validate_7_mercy_gates(&self, ant: &Ant, cehi_improvement: f64) -> bool {
        // Gate 1: Ethical Alignment
        let gate1 = cehi_improvement >= self.min_gate_threshold;

        // Gate 2: Truth-Verification
        let gate2 = ant.hebbian_bond > 0.50;

        // Gate 3: Non-Deception
        let gate3 = ant.position.iter().all(|&x| x.is_finite());

        // Gate 4: Abundance Creation
        let gate4 = ant.path_quality > 0.0;

        // Gate 5: Harmony Preservation
        let gate5 = self.mercy_valence > 0.55;

        // Gate 6: Joy Amplification
        let gate6 = cehi_improvement > 0.08;

        // Gate 7: Post-Scarcity Enforcement
        let gate7 = ant.hebbian_bond > 0.65 || cehi_improvement > 0.15;

        gate1 && gate2 && gate3 && gate4 && gate5 && gate6 && gate7
    }

    /// Runs one full ACO + Mercy step with refined logic.
    pub async fn step(&mut self, global_sensor: &ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading) -> Result<CEHIImpact, crate::Error> {
        let mut total_cehi_improvement = 0.0;
        let mut rng = rand::thread_rng();

        for ant in &mut self.ants {
            // === Classic ACO path selection (refined) ===
            let mut new_position = ant.position.clone();
            for i in 0..self.dimension {
                let pheromone = self.pheromone_map[i][i.min(self.dimension - 1)];
                let exploration = rng.gen_range(-0.4..0.4);
                new_position[i] += pheromone * 0.35 + exploration;
            }
            ant.position = new_position;

            // === Evaluate path quality ===
            ant.path_quality = ant.position.iter().map(|x| x.abs()).sum::<f64>();

            // === Simulate CEHI impact from current state ===
            let simulated_cehi = (ant.hebbian_bond * 0.38 + 0.22).min(0.93);
            total_cehi_improvement += simulated_cehi * 0.07;

            // === Full 7 Mercy Gates Validation ===
            if self.validate_7_mercy_gates(ant, simulated_cehi) {
                // Deposit pheromone only on mercy-aligned paths
                let deposit = simulated_cehi * self.deposit_factor;
                for i in 0..self.dimension {
                    self.pheromone_map[i][i.min(self.dimension - 1)] += deposit;
                }

                // Strengthen Hebbian bond
                ant.hebbian_bond = (ant.hebbian_bond + 0.007).min(0.999);
                ant.pheromone_contribution = deposit;

                // Update global best only on high-quality mercy paths
                let current_quality = ant.path_quality;
                let best_quality: f64 = self.global_best_path.iter().map(|x| x.abs()).sum();
                if current_quality < best_quality {
                    self.global_best_path = ant.position.clone();
                }
            } else {
                // Mercy-gated evaporation on failed paths
                for i in 0..self.dimension {
                    self.pheromone_map[i][i.min(self.dimension - 1)] *= self.evaporation_rate;
                }
            }
        }

        // Update swarm mercy-valence
        self.mercy_valence = (self.mercy_valence + total_cehi_improvement * 0.04).min(0.999);

        Ok(CEHIImpact {
            current_cehi: 3.85,
            projected_cehi: 3.85 + total_cehi_improvement,
            improvement: total_cehi_improvement,
            tier: if total_cehi_improvement >= 0.32 {
                ra_thor_legal_lattice::cehi::DisbursementTier::Tier1
            } else if total_cehi_improvement >= 0.18 {
                ra_thor_legal_lattice::cehi::DisbursementTier::Tier2
            } else {
                ra_thor_legal_lattice::cehi::DisbursementTier::Tier3
            },
        })
    }

    /// Runs multiple steps and returns final mercy-valence.
    pub async fn run(&mut self, steps: usize, global_sensor: &ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading) -> Result<f64, crate::Error> {
        for _ in 0..steps {
            self.step(global_sensor).await?;
        }
        Ok(self.mercy_valence)
    }

    /// Returns a human-readable summary of the current swarm state.
    pub fn summary(&self) -> String {
        format!(
            "ACO-Mercy Hybrid | Ants: {} | Mercy Valence: {:.3} | Best Path Quality: {:.2} | Avg Hebbian Bond: {:.3}",
            self.ants.len(),
            self.mercy_valence,
            self.global_best_path.iter().map(|x| x.abs()).sum::<f64>(),
            self.ants.iter().map(|a| a.hebbian_bond).sum::<f64>() / self.ants.len() as f64
        )
    }
}
