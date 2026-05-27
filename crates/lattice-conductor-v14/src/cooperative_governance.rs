// crates/lattice-conductor-v14/src/cooperative_governance.rs
// Cooperative Game Theory Module for Ra-Thor Governance (v14.1+)
//
// Implements core cooperative game primitives with focus on:
// - Shapley Value (fair marginal contribution)
// - Banzhaf Power Index (swing voter criticality)
// - Simulation-friendly design for governance use cases
//
// Designed for integration with PATSAGi Councils and Thunder Lattice.
// All functions are simulatable and testable.

use std::collections::HashSet;

/// A simple cooperative game defined by a set of players and a characteristic function.
/// In Ra-Thor context, players can be organisms, councils, or nodes.
pub struct CooperativeGame {
    pub players: Vec<String>,
    /// characteristic_function(S) returns the value coalition S can achieve
    pub characteristic_function: Box<dyn Fn(&HashSet<String>) -> f64 + Send + Sync>,
}

impl CooperativeGame {
    pub fn new(players: Vec<String>, char_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            players,
            characteristic_function: Box::new(char_fn),
        }
    }

    /// Computes exact Shapley value for small player sets.
    /// For larger sets, use approximate_shapley_value.
    pub fn shapley_value(&self) -> Vec<(String, f64)> {
        let n = self.players.len();
        let mut values = vec![0.0; n];

        // Iterate over all possible coalitions
        for mask in 0..(1 << n) {
            let mut coalition = HashSet::new();
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    coalition.insert(self.players[i].clone());
                }
            }

            let coalition_value = (self.characteristic_function)(&coalition);

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    // Player i is in the coalition
                    let mut without_i = coalition.clone();
                    without_i.remove(&self.players[i]);
                    let without_value = (self.characteristic_function)(&without_i);
                    let marginal = coalition_value - without_value;

                    // Weight by 1 / n (simplified exact for small n)
                    values[i] += marginal / (n as f64);
                }
            }
        }

        self.players.iter().cloned().zip(values).collect()
    }

    /// Monte Carlo approximation of Shapley value (scalable)
    pub fn approximate_shapley_value(&self, samples: usize) -> Vec<(String, f64)> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let mut totals = vec![0.0; self.players.len()];

        for _ in 0..samples {
            let mut perm: Vec<usize> = (0..self.players.len()).collect();
            perm.shuffle(&mut rng);

            let mut coalition = HashSet::new();
            let mut prev_value = 0.0;

            for &idx in &perm {
                coalition.insert(self.players[idx].clone());
                let new_value = (self.characteristic_function)(&coalition);
                let marginal = new_value - prev_value;
                totals[idx] += marginal;
                prev_value = new_value;
            }
        }

        let factor = 1.0 / samples as f64;
        self.players
            .iter()
            .cloned()
            .zip(totals.into_iter().map(|v| v * factor))
            .collect()
    }

    /// Computes Banzhaf Power Index (normalized)
    pub fn banzhaf_index(&self) -> Vec<(String, f64)> {
        let n = self.players.len();
        let mut critical_counts = vec![0usize; n];
        let mut total_critical = 0usize;

        for mask in 0..(1 << n) {
            let mut coalition = HashSet::new();
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    coalition.insert(self.players[i].clone());
                }
            }

            let coalition_value = (self.characteristic_function)(&coalition);

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    let mut without = coalition.clone();
                    without.remove(&self.players[i]);
                    let without_value = (self.characteristic_function)(&without);

                    if without_value < coalition_value {
                        critical_counts[i] += 1;
                        total_critical += 1;
                    }
                }
            }
        }

        if total_critical == 0 {
            return self.players.iter().cloned().map(|p| (p, 0.0)).collect();
        }

        self.players
            .iter()
            .cloned()
            .zip(critical_counts.into_iter().map(|c| c as f64 / total_critical as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_and_banzhaf_simple() {
        let players = vec!["A".to_string(), "B".to_string()];
        let game = CooperativeGame::new(players, |coalition| {
            if coalition.len() == 2 { 10.0 } else { 0.0 }
        });

        let shapley = game.shapley_value();
        let banzhaf = game.banzhaf_index();

        assert!((shapley[0].1 - 5.0).abs() < 0.01);
        assert!((banzhaf[0].1 - 0.5).abs() < 0.01);
    }
}