// crates/lattice-conductor-v14/src/cooperative_governance.rs
// Cooperative Game Theory + Shapley Optimization (Improved)

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

    pub fn shapley_value(&self) -> Vec<(String, f64)> { /* ... existing ... */ 
        // (keeping previous implementation)
        let n = self.players.len();
        let mut values = vec![0.0; n];
        for mask in 0..(1 << n) {
            let mut coalition = HashSet::new();
            for i in 0..n { if (mask & (1 << i)) != 0 { coalition.insert(self.players[i].clone()); } }
            let v = (self.characteristic_function)(&coalition);
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    let mut w = coalition.clone(); w.remove(&self.players[i]);
                    values[i] += v - (self.characteristic_function)(&w);
                }
            }
        }
        self.players.iter().cloned().zip(values.into_iter().map(|v| v / n as f64)).collect()
    }

    pub fn approximate_shapley_value(&self, samples: usize) -> Vec<(String, f64)> { /* ... */ 
        // (keeping previous)
        use rand::seq::SliceRandom; use rand::thread_rng;
        let mut rng = thread_rng();
        let mut totals = vec![0.0; self.players.len()];
        for _ in 0..samples {
            let mut perm: Vec<usize> = (0..self.players.len()).collect(); perm.shuffle(&mut rng);
            let mut coalition = HashSet::new(); let mut prev = 0.0;
            for &i in &perm {
                coalition.insert(self.players[i].clone());
                let new_v = (self.characteristic_function)(&coalition);
                totals[i] += new_v - prev; prev = new_v;
            }
        }
        let f = 1.0 / samples as f64;
        self.players.iter().cloned().zip(totals.into_iter().map(|v| v * f)).collect()
    }

    pub fn banzhaf_index(&self) -> Vec<(String, f64)> { /* ... existing ... */ }

    // ==================== Improved Shapley Optimization ====================

    /// Improved optimizer with random restarts for better exploration
    pub fn optimize_coalition_for_fair_shapley(&self, max_size: usize, restarts: usize) -> (Vec<String>, f64) {
        let mut best_coalition = vec![];
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..restarts.max(1) {
            let (coal, score) = self.greedy_fair_shapley(max_size);
            if score > best_score {
                best_score = score;
                best_coalition = coal;
            }
        }
        (best_coalition, best_score.max(0.0))
    }

    fn greedy_fair_shapley(&self, max_size: usize) -> (Vec<String>, f64) {
        let mut current = HashSet::new();
        let mut remaining: Vec<String> = self.players.clone();
        let mut best_min = f64::NEG_INFINITY;
        let mut best_set = vec![];

        for _ in 0..max_size.min(self.players.len()) {
            let mut best_cand: Option<String> = None;
            let mut best_score = f64::NEG_INFINITY;

            for cand in &remaining {
                let mut test = current.clone(); test.insert(cand.clone());
                let temp = CooperativeGame::new(test.iter().cloned().collect(), |s| (self.characteristic_function)(s));
                let shap = temp.shapley_value();
                let min_shap = shap.iter().map(|(_,v)| *v).fold(f64::INFINITY, f64::min);

                if min_shap > best_score {
                    best_score = min_shap;
                    best_cand = Some(cand.clone());
                }
            }

            if let Some(c) = best_cand {
                current.insert(c); remaining.retain(|p| p != &c);
                if best_score > best_min { best_min = best_score; best_set = current.iter().cloned().collect(); }
            } else { break; }
        }
        (best_set, best_min)
    }
}

#[cfg(test)]
mod tests { use super::*; /* ... */ }
