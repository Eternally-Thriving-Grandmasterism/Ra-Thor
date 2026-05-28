//! Governance & Cooperative Game Theory Layer
//!
//! Monte Carlo Shapley approximation + Influence-based value function.
//! Built with accuracy and scalability in mind.

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

/// Value function backed by real Phase 4 InfluenceScore.
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
                if let GovernancePlayer::Claim(claim_id) = p {
                    Some(self.graph.calculate_influence_score(*claim_id).total)
                } else {
                    None
                }
            })
            .sum()
    }
}

/// Shapley Value Calculator with Monte Carlo approximation support.
pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

    /// Monte Carlo approximation of Shapley values.
    /// Samples random permutations and estimates marginal contributions.
    pub fn calculate_monte_carlo(
        &self,
        players: &[GovernancePlayer],
        samples: usize,
    ) -> HashMap<GovernancePlayer, f64> {
        if players.is_empty() || samples == 0 {
            return HashMap::new();
        }

        let mut sums: HashMap<GovernancePlayer, f64> = HashMap::new();
        let mut rng = thread_rng();

        for _ in 0..samples {
            let mut permutation = players.to_vec();
            permutation.shuffle(&mut rng);

            let mut coalition = Vec::new();
            let mut prev_value = 0.0;

            for player in permutation {
                coalition.push(player.clone());
                let new_value = self.value_function.coalition_value(&coalition);
                let marginal = new_value - prev_value;

                *sums.entry(player).or_insert(0.0) += marginal;
                prev_value = new_value;
            }
        }

        // Average over number of samples
        sums.into_iter()
            .map(|(player, total)| (player, total / samples as f64))
            .collect()
    }
}

/// Convenience constructor using real influence data.
pub fn influence_shapley_calculator(graph: &ArgumentGraph) -> ShapleyValueCalculator {
    ShapleyValueCalculator::new(Box::new(InfluenceBasedValueFunction::new(graph)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_empty() {
        let calc = ShapleyValueCalculator::new(Box::new(DefaultValueFunction));
        let values = calc.calculate_monte_carlo(&[], 100);
        assert!(values.is_empty());
    }

    #[test]
    fn test_monte_carlo_basic() {
        let mut graph = ArgumentGraph::new();
        let c1 = graph.add_claim("C1".to_string(), "Test".to_string(), 0.8);
        let c2 = graph.add_claim("C2".to_string(), "Test".to_string(), 0.7);

        let calc = influence_shapley_calculator(&graph);
        let players = vec![GovernancePlayer::Claim(c1), GovernancePlayer::Claim(c2)];

        let values = calc.calculate_monte_carlo(&players, 50);
        assert_eq!(values.len(), 2);
    }
}
