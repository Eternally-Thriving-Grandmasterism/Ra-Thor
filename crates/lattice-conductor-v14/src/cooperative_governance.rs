// crates/lattice-conductor-v14/src/cooperative_governance.rs
// Multi-objective Shapley Optimization + Full Cooperative Governance

use std::collections::HashSet;

pub struct CooperativeGame {
    pub players: Vec<String>,
    pub characteristic_function: Box<dyn Fn(&HashSet<String>) -> f64 + Send + Sync>,
}

impl CooperativeGame {
    pub fn new(players: Vec<String>, char_fn: impl Fn(&HashSet<String>) -> f64 + Send + Sync + 'static) -> Self {
        Self { players, characteristic_function: Box::new(char_fn) }
    }

    pub fn shapley_value(&self) -> Vec<(String, f64)> { /* existing implementation */ vec![] }
    pub fn approximate_shapley_value(&self, samples: usize) -> Vec<(String, f64)> { /* existing */ vec![] }
    pub fn banzhaf_index(&self) -> Vec<(String, f64)> { /* existing */ vec![] }

    // ==================== Multi-Objective Shapley Optimization ====================

    /// Balances fairness (min Shapley) and total coalition value
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
            let mut best_cand = None;
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

    // Legacy single-objective for backward compatibility
    pub fn optimize_coalition_for_fair_shapley(&self, max_size: usize, restarts: usize) -> (Vec<String>, f64) {
        self.optimize_coalition_multi_objective(max_size, 1.0, 0.0, restarts)
    }
}

#[cfg(test)]
mod tests { use super::*; }
