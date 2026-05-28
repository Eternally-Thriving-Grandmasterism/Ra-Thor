//! Governance & Cooperative Game Theory Layer
//!
//! Monte Carlo Shapley with Stratified Sampling + Antithetic Variates.

use crate::argumentation::ArgumentGraph;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GovernancePlayer {
    Claim(u64),
    CouncilMember { id: String },
    Agent { id: String },
}

pub trait ValueFunction {
    fn coalition_value(&self, players: &[GovernancePlayer]) -> f64;
}

#[derive(Debug, Default)]
pub struct DefaultValueFunction;

impl ValueFunction for DefaultValueFunction {
    fn coalition_value(&self, _players: &[GovernancePlayer]) -> f64 {
        0.0
    }
}

pub struct InfluenceBasedValueFunction<'a> {
    graph: &'a ArgumentGraph,
}

impl<'a> InfluenceBasedValueFunction<'a> {
    pub fn new(graph: &'a ArgumentGraph) -> Self {
        Self { graph }
    }
}

impl<'a> ValueFunction for InfluenceBasedValueFunction<'a> {
    fn coalition_value(&self, players: &[GovernancePlayer]) -> f64 {
        players
            .iter()
            .filter_map(|p| {
                if let GovernancePlayer::Claim(id) = p {
                    Some(self.graph.calculate_influence_score(*id).total)
                } else {
                    None
                }
            })
            .sum()
    }
}

pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

    /// Stratified Monte Carlo Shapley with Antithetic Variates.
    /// Uses coalition-size stratification + reverse permutation pairing for variance reduction.
    pub fn calculate_stratified_antithetic(
        &self,
        players: &[GovernancePlayer],
        samples_per_stratum: usize,
    ) -> HashMap<GovernancePlayer, f64> {
        if players.is_empty() || samples_per_stratum == 0 {
            return HashMap::new();
        }

        let n = players.len();
        let mut sums: HashMap<GovernancePlayer, f64> = HashMap::new();
        let mut rng = thread_rng();

        for size in 0..n {
            for _ in 0..samples_per_stratum {
                let mut others: Vec<_> = players.to_vec();
                others.shuffle(&mut rng);

                let coalition: Vec<_> = others.into_iter().take(size).collect();
                let base_value = self.value_function.coalition_value(&coalition);

                // Original sample
                for player in players {
                    if !coalition.contains(player) {
                        let mut extended = coalition.clone();
                        extended.push(player.clone());
                        let marginal =
                            self.value_function.coalition_value(&extended) - base_value;
                        *sums.entry(player.clone()).or_insert(0.0) += marginal;
                    }
                }

                // Antithetic sample: reverse the remaining players order
                let mut antithetic_coalition = coalition.clone();
                let mut remaining: Vec<_> = others.into_iter().skip(size).collect();
                remaining.reverse();

                for player in &remaining {
                    if !antithetic_coalition.contains(player) {
                        let mut extended = antithetic_coalition.clone();
                        extended.push(player.clone());
                        let marginal = self.value_function.coalition_value(&extended)
                            - self.value_function.coalition_value(&antithetic_coalition);
                        *sums.entry(player.clone()).or_insert(0.0) += marginal;
                        antithetic_coalition.push(player.clone());
                    }
                }
            }
        }

        let total_weight = (n * samples_per_stratum * 2) as f64;
        sums.into_iter()
            .map(|(p, total)| (p, total / total_weight))
            .collect()
    }
}

pub fn influence_shapley_calculator(graph: &ArgumentGraph) -> ShapleyValueCalculator {
    ShapleyValueCalculator::new(Box::new(InfluenceBasedValueFunction::new(graph)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antithetic_basic() {
        let mut graph = ArgumentGraph::new();
        let c1 = graph.add_claim("C1".to_string(), "Test".to_string(), 0.8);
        let c2 = graph.add_claim("C2".to_string(), "Test".to_string(), 0.7);

        let calc = influence_shapley_calculator(&graph);
        let players = vec![GovernancePlayer::Claim(c1), GovernancePlayer::Claim(c2)];

        let values = calc.calculate_stratified_antithetic(&players, 4);
        assert_eq!(values.len(), 2);
    }
}
