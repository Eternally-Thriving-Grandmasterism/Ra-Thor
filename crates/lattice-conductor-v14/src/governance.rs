//! Governance & Cooperative Game Theory Layer
//!
//! Monte Carlo Shapley with Stratified Sampling (variance reduction).

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

    /// Monte Carlo Shapley with stratified sampling by coalition size.
    /// Reduces variance compared to pure random sampling.
    pub fn calculate_stratified_monte_carlo(
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

        // Stratify by coalition size (0 to n-1 players before the target player)
        for size in 0..n {
            for _ in 0..samples_per_stratum {
                // Sample a random coalition of exact size `size`
                let mut others: Vec<_> = players.to_vec();
                others.shuffle(&mut rng);

                let coalition: Vec<_> = others.into_iter().take(size).collect();
                let base_value = self.value_function.coalition_value(&coalition);

                // For each player not in the coalition, compute marginal contribution
                for player in players {
                    if !coalition.contains(player) {
                        let mut extended = coalition.clone();
                        extended.push(player.clone());

                        let new_value = self.value_function.coalition_value(&extended);
                        let marginal = new_value - base_value;

                        *sums.entry(player.clone()).or_insert(0.0) += marginal;
                    }
                }
            }
        }

        // Normalize
        let total_samples = (n * samples_per_stratum) as f64;
        sums.into_iter()
            .map(|(p, total)| (p, total / total_samples))
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
    fn test_stratified_empty() {
        let calc = ShapleyValueCalculator::new(Box::new(DefaultValueFunction));
        let values = calc.calculate_stratified_monte_carlo(&[], 10);
        assert!(values.is_empty());
    }

    #[test]
    fn test_stratified_basic() {
        let mut graph = ArgumentGraph::new();
        let c1 = graph.add_claim("C1".to_string(), "Test".to_string(), 0.8);
        let c2 = graph.add_claim("C2".to_string(), "Test".to_string(), 0.7);

        let calc = influence_shapley_calculator(&graph);
        let players = vec![GovernancePlayer::Claim(c1), GovernancePlayer::Claim(c2)];

        let values = calc.calculate_stratified_monte_carlo(&players, 5);
        assert_eq!(values.len(), 2);
    }
}
