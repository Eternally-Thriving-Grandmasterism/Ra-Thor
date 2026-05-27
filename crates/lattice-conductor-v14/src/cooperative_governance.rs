// crates/lattice-conductor-v14/src/cooperative_governance.rs
// Cooperative Game Theory + Shapley Value Optimization

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

    pub fn shapley_value(&self) -> Vec<(String, f64)> {
        let n = self.players.len();
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
                    values[i] += marginal / (n as f64);
                }
            }
        }

        self.players.iter().cloned().zip(values).collect()
    }

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

        self.players.iter().cloned().zip(critical_counts.into_iter().map(|c| c as f64 / total_critical as f64)).collect()
    }

    // ==================== Shapley Value Optimization ====================

    /// Finds a subset of players that maximizes the minimum Shapley value (fairness-focused)
    /// Uses a simple greedy + local search approach for tractability.
    pub fn optimize_coalition_for_fair_shapley(&self, max_size: usize) -> (Vec<String>, f64) {
        let n = self.players.len();
        if n == 0 { return (vec![], 0.0); }

        let mut best_coalition: Vec<String> = vec![];
        let mut best_min_shapley = f64::NEG_INFINITY;

        // Greedy seed: start with highest individual contributors
        let mut current: HashSet<String> = HashSet::new();
        let mut remaining: Vec<String> = self.players.clone();

        // Add players one by one while improving min Shapley
        for _ in 0..max_size.min(n) {
            let mut best_candidate: Option<String> = None;
            let mut best_score = f64::NEG_INFINITY;

            for candidate in &remaining {
                let mut test_set = current.clone();
                test_set.insert(candidate.clone());

                let temp_game = CooperativeGame::new(
                    test_set.iter().cloned().collect(),
                    |s| (self.characteristic_function)(s),
                );
                let shapley = temp_game.shapley_value();
                let min_shapley = shapley.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);

                if min_shapley > best_score {
                    best_score = min_shapley;
                    best_candidate = Some(candidate.clone());
                }
            }

            if let Some(best) = best_candidate {
                current.insert(best.clone());
                remaining.retain(|p| p != &best);

                if best_score > best_min_shapley {
                    best_min_shapley = best_score;
                    best_coalition = current.iter().cloned().collect();
                }
            } else {
                break;
            }
        }

        (best_coalition, best_min_shapley.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_coalition() {
        let players = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let game = CooperativeGame::new(players, |s| s.len() as f64 * 10.0);

        let (best, score) = game.optimize_coalition_for_fair_shapley(3);
        assert!(!best.is_empty());
        assert!(score >= 0.0);
    }
}