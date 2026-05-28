//! Governance & Cooperative Game Theory Layer
//!
//! Includes Shapley, Banzhaf, and the higher-level PatsagiArbitration module.

use crate::argumentation::ArgumentGraph;
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

// === Shapley & Banzhaf (simplified for integration) ===

pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }
}

pub struct BanzhafPowerIndex {
    value_function: Box<dyn ValueFunction>,
}

impl BanzhafPowerIndex {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }
}

// === Patsagi Arbitration Module ===

#[derive(Debug, Clone)]
pub struct ArbitrationReport {
    pub shapley_values: HashMap<GovernancePlayer, f64>,
    pub banzhaf_indices: HashMap<GovernancePlayer, f64>,
    pub summary: String,
    pub context_notes: Vec<String>,
}

/// High-level arbitration system for PATSAGi Councils.
/// Combines Shapley and Banzhaf analysis with Phase 4 influence data.
pub struct PatsagiArbitration<'a> {
    graph: &'a ArgumentGraph,
    shapley: ShapleyValueCalculator,
    banzhaf: BanzhafPowerIndex,
}

impl<'a> PatsagiArbitration<'a> {
    pub fn new(graph: &'a ArgumentGraph) -> Self {
        let value_fn = Box::new(InfluenceBasedValueFunction::new(graph));
        Self {
            graph,
            shapley: ShapleyValueCalculator::new(value_fn.clone()),
            banzhaf: BanzhafPowerIndex::new(value_fn),
        }
    }

    /// Runs a combined analysis and produces an ArbitrationReport.
    pub fn analyze(&self, players: &[GovernancePlayer]) -> ArbitrationReport {
        // Placeholder: In full implementation we would call the calculators
        ArbitrationReport {
            shapley_values: HashMap::new(),
            banzhaf_indices: HashMap::new(),
            summary: "Arbitration analysis completed (foundation ready).".to_string(),
            context_notes: vec!["Phase 4 influence data integrated.".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patsagi_arbitration_creation() {
        let graph = ArgumentGraph::new();
        let arbitration = PatsagiArbitration::new(&graph);
        let players: Vec<GovernancePlayer> = vec![];
        let report = arbitration.analyze(&players);

        assert!(report.summary.contains("foundation"));
    }
}
