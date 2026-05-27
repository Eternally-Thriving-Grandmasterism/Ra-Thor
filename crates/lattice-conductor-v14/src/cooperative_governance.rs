// crates/lattice-conductor-v14/src/cooperative_governance.rs
// Cooperative Game Theory Module for Ra-Thor (Restorative Merge v14.1)
//
// This file restores full previous working implementations of Shapley and Banzhaf,
// then cleanly layers multi-objective optimization on top.
// Restorative merge performed to recover lost core logic.

use std::collections::HashSet;

pub struct CooperativeGame {
    pub players: Vec<String>,
    pub characteristic_function: Box<dyn Fn(&HashSet<String>) -> f64 + Send + Sync>,
}

impl CooperativeGame {
    pub fn new(players: Vec<String>, char_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            players,
            characteristic_function: Box::new(char_fn),
        }
    }

    /// Exact Shapley value calculation (for small n)
    pub fn shapley_value(&self) -> Vec<(String, f64)> {
        let n = self.players.len();
        if n == 0 { return vec![]; }

        let mut values = vec![0.0; n];

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
                    let mut without_i = coalition.clone();
                    without_i.remove(&self.players[i]);
                    let without_value = (self.characteristic_function)(&without_i);
                    let marginal = coalition_value - without_value;
                    values[i] += marginal;
                }
            }
        }

        let factor = 1.0 / n as f64;
        self.players.iter().cloned().zip(values.into_iter().map(|v| v * factor)).collect()
    }

    /// Monte Carlo approximation of Shapley value
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
        self.players.iter().cloned().zip(totals.into_iter().map(|v| v * factor)).collect()
    }

    /// Banzhaf Power Index (normalized)
    pub fn banzhaf_index(&self) -> Vec<(String, f64)> {
        let n = self.players.len();
        if n == 0 { return vec![]; }

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

        self.players.iter().cloned().zip(critical_counts.into_iter().map(|c| c as f64 / total_critical as f64)).collect()
    }

    // ==================== Multi-Objective Shapley Optimization ====================

    pub fn optimize_coalition_multi_objective(
        &self,
        max_size: usize,
        fairness_weight: f64,
        value_weight: f64,
        restarts: usize,
    ) -> (Vec<String>, f64) {
        let mut best_coalition = vec![];
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..restarts.max(1) {
            let (coal, score) = self.greedy_multi_objective(max_size, fairness_weight, value_weight);
            if score > best_score {
                best_score = score;
                best_coalition = coal;
            }
        }
        (best_coalition, best_score)
    }

    fn greedy_multi_objective(&self, max_size: usize, fairness_w: f64, value_w: f64) -> (Vec<String>, f64) {
        let mut current = HashSet::new();
        let mut remaining = self.players.clone();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_set = vec![];

        for _ in 0..max_size.min(self.players.len()) {
            let mut best_cand: Option<String> = None;
            let mut best_local = f64::NEG_INFINITY;

            for cand in &remaining {
                let mut test = current.clone();
                test.insert(cand.clone());

                let temp_game = CooperativeGame::new(test.iter().cloned().collect(), |s| (self.characteristic_function)(s));
                let shapley = temp_game.shapley_value();

                let min_shapley = shapley.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
                let total_value = (self.characteristic_function)(&test);

                let score = fairness_w * min_shapley + value_w * total_value;

                if score > best_local {
                    best_local = score;
                    best_cand = Some(cand.clone());
                }
            }

            if let Some(c) = best_cand {
                current.insert(c.clone());
                remaining.retain(|p| p != &c);
                if best_local > best_score {
                    best_score = best_local;
                    best_set = current.iter().cloned().collect();
                }
            } else {
                break;
            }
        }
        (best_set, best_score)
    }

    // Backward compatible single-objective version
    pub fn optimize_coalition_for_fair_shapley(&self, max_size: usize, restarts: usize) -> (Vec<String>, f64) {
        self.optimize_coalition_multi_objective(max_size, 1.0, 0.0, restarts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_and_optimization() {
        let players = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let game = CooperativeGame::new(players, |s| s.len() as f64 * 10.0);

        let shapley = game.shapley_value();
        assert_eq!(shapley.len(), 3);

        let (best, _) = game.optimize_coalition_for_fair_shapley(2, 3);
        assert!(!best.is_empty());
    }
}
