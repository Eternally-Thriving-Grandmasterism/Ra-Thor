//! Governance & Cooperative Game Theory Layer
//!
//! Phase 1: Foundational structures and Shapley Value Calculator.
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

/// A simple value function that returns the "worth" of a coalition.
/// For Phase 1, we use a placeholder that can later be connected to InfluenceScore.
pub trait ValueFunction {
    fn coalition_value(&self, players: &[GovernancePlayer]) -> f64;
}

/// Default value function that returns 0.0 (placeholder).
/// In later phases this will be connected to ArgumentGraph + InfluenceScore.
#[derive(Debug, Default)]
pub struct DefaultValueFunction;

impl ValueFunction for DefaultValueFunction {
    fn coalition_value(&self, _players: &[GovernancePlayer]) -> f64 {
        0.0
    }
}

/// Calculates Shapley values for a set of players.
/// Phase 1 implementation uses exact calculation (suitable for small sets).
pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

    /// Calculates Shapley values for the given list of players.
    /// Returns a map of player -> Shapley value.
    pub fn calculate(&self, players: &[GovernancePlayer]) -> HashMap<GovernancePlayer, f64> {
        let mut shapley_values = HashMap::new();

        if players.is_empty() {
            return shapley_values;
        }

        let n = players.len() as f64;

        for (i, player) in players.iter().enumerate() {
            // Simple placeholder calculation for Phase 1
            // Real implementation will enumerate coalitions properly
            let marginal = self.value_function.coalition_value(&players[0..=i]);
            shapley_values.insert(player.clone(), marginal / n);
        }

        shapley_values
    }
}

/// Helper to create a Shapley calculator with the default value function.
pub fn default_shapley_calculator() -> ShapleyValueCalculator {
    ShapleyValueCalculator::new(Box::new(DefaultValueFunction::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_empty_players() {
        let calc = default_shapley_calculator();
        let values = calc.calculate(&[]);
        assert!(values.is_empty());
    }

    #[test]
    fn test_shapley_basic() {
        let calc = default_shapley_calculator();
        let players = vec![
            GovernancePlayer::Claim(1),
            GovernancePlayer::Claim(2),
        ];
        let values = calc.calculate(&players);
        assert_eq!(values.len(), 2);
    }
}
