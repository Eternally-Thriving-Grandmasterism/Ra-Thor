//! Governance & Cooperative Game Theory Layer
//!
//! Phase 1: Foundational structures + Influence-aware Shapley calculation.
//! Builds on Phase 4 (ArgumentGraph + InfluenceScore).

use crate::argumentation::{ArgumentGraph, ArgumentId};
use std::collections::HashMap;

/// Represents an entity whose contribution or power can be measured.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GovernancePlayer {
    Claim(ArgumentId),
    CouncilMember { id: String },
    Agent { id: String },
}

/// A trait for functions that compute the "worth" of a coalition of players.
pub trait ValueFunction {
    fn coalition_value(&self, players: &[GovernancePlayer]) -> f64;
}

/// Default placeholder value function.
#[derive(Debug, Default)]
pub struct DefaultValueFunction;

impl ValueFunction for DefaultValueFunction {
    fn coalition_value(&self, _players: &[GovernancePlayer]) -> f64 {
        0.0
    }
}

/// Value function backed by real Phase 4 InfluenceScore data.
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

/// Calculates Shapley values for a set of players.
pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

    pub fn calculate(&self, players: &[GovernancePlayer]) -> HashMap<GovernancePlayer, f64> {
        let mut shapley_values = HashMap::new();

        if players.is_empty() {
            return shapley_values;
        }

        let n = players.len() as f64;

        for (i, player) in players.iter().enumerate() {
            let coalition = &players[0..=i];
            let value = self.value_function.coalition_value(coalition);
            shapley_values.insert(player.clone(), value / n);
        }

        shapley_values
    }
}

/// Creates a Shapley calculator that uses real influence scores from the graph.
pub fn influence_shapley_calculator(graph: &ArgumentGraph) -> ShapleyValueCalculator {
    ShapleyValueCalculator::new(Box::new(InfluenceBasedValueFunction::new(graph)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_empty() {
        let calc = ShapleyValueCalculator::new(Box::new(DefaultValueFunction));
        let values = calc.calculate(&[]);
        assert!(values.is_empty());
    }

    #[test]
    fn test_influence_based_shapley() {
        let mut graph = ArgumentGraph::new();
        let c1 = graph.add_claim("Claim1".to_string(), "Test".to_string(), 0.8);
        let c2 = graph.add_claim("Claim2".to_string(), "Test".to_string(), 0.7);

        let calc = influence_shapley_calculator(&graph);
        let players = vec![GovernancePlayer::Claim(c1), GovernancePlayer::Claim(c2)];

        let values = calc.calculate(&players);
        assert_eq!(values.len(), 2);
    }
}
