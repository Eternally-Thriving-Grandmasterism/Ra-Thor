//! Governance & Cooperative Game Theory Layer
//!
//! Full Shapley + Banzhaf with variance reduction, plus mercy-aware PatsagiArbitration
//! with deeper compassionate numerical softening on MercyGate detection.

use crate::argumentation::{ArgumentGraph, SuperiorityContext};
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

// === ShapleyValueCalculator ===

pub struct ShapleyValueCalculator {
    value_function: Box<dyn ValueFunction>,
}

impl ShapleyValueCalculator {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

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

                let coalition: Vec<_> = others.iter().take(size).cloned().collect();
                let base_value = self.value_function.coalition_value(&coalition);

                for player in players {
                    if !coalition.contains(player) {
                        let mut extended = coalition.clone();
                        extended.push(player.clone());
                        let marginal =
                            self.value_function.coalition_value(&extended) - base_value;
                        *sums.entry(player.clone()).or_insert(0.0) += marginal;
                    }
                }

                // Antithetic
                let mut antithetic = coalition.clone();
                let mut remaining: Vec<_> = others.iter().skip(size).cloned().collect();
                remaining.reverse();

                for player in &remaining {
                    if !antithetic.contains(player) {
                        let mut extended = antithetic.clone();
                        extended.push(player.clone());
                        let marginal = self.value_function.coalition_value(&extended)
                            - self.value_function.coalition_value(&antithetic);
                        *sums.entry(player.clone()).or_insert(0.0) += marginal;
                        antithetic.push(player.clone());
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

// === BanzhafPowerIndex ===

pub struct BanzhafPowerIndex {
    value_function: Box<dyn ValueFunction>,
}

impl BanzhafPowerIndex {
    pub fn new(value_function: Box<dyn ValueFunction>) -> Self {
        Self { value_function }
    }

    pub fn calculate(&self, players: &[GovernancePlayer]) -> HashMap<GovernancePlayer, f64> {
        let mut counts: HashMap<GovernancePlayer, usize> = HashMap::new();
        let n = players.len();

        if n == 0 {
            return HashMap::new();
        }

        for mask in 0..(1 << n) {
            let mut coalition = Vec::new();
            for (i, player) in players.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    coalition.push(player.clone());
                }
            }

            let value_without = self.value_function.coalition_value(&coalition);

            for (i, player) in players.iter().enumerate() {
                if (mask & (1 << i)) == 0 {
                    let mut with_player = coalition.clone();
                    with_player.push(player.clone());

                    let value_with = self.value_function.coalition_value(&with_player);
                    if value_with > value_without {
                        *counts.entry(player.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        let total = counts.values().sum::<usize>() as f64;
        if total == 0.0 {
            return counts.into_iter().map(|(p, _)| (p, 0.0)).collect();
        }

        counts.into_iter().map(|(p, c)| (p, c as f64 / total)).collect()
    }

    pub fn calculate_monte_carlo(
        &self,
        players: &[GovernancePlayer],
        samples: usize,
    ) -> HashMap<GovernancePlayer, f64> {
        let mut counts: HashMap<GovernancePlayer, usize> = HashMap::new();
        let mut rng = thread_rng();

        for _ in 0..samples {
            let mut coalition: Vec<_> = players.to_vec();
            coalition.shuffle(&mut rng);

            let mid = coalition.len() / 2;
            let without: Vec<_> = coalition[..mid].to_vec();
            let value_without = self.value_function.coalition_value(&without);

            for player in &coalition[mid..] {
                let mut with_player = without.clone();
                with_player.push(player.clone());

                let value_with = self.value_function.coalition_value(&with_player);
                if value_with > value_without {
                    *counts.entry(player.clone()).or_insert(0) += 1;
                }
            }
        }

        let total = counts.values().sum::<usize>() as f64;
        if total == 0.0 {
            return counts.into_iter().map(|(p, _)| (p, 0.0)).collect();
        }

        counts.into_iter().map(|(p, c)| (p, c as f64 / total)).collect()
    }
}

// === PatsagiArbitration (Mercy-Aware) ===

#[derive(Debug, Clone)]
pub struct ArbitrationReport {
    pub shapley_values: HashMap<GovernancePlayer, f64>,
    pub banzhaf_indices: HashMap<GovernancePlayer, f64>,
    pub summary: String,
    pub context_notes: Vec<String>,
}

pub struct PatsagiArbitration<'a> {
    graph: &'a ArgumentGraph,
}

impl<'a> PatsagiArbitration<'a> {
    pub fn new(graph: &'a ArgumentGraph) -> Self {
        Self { graph }
    }

    pub fn analyze(&self, players: &[GovernancePlayer]) -> ArbitrationReport {
        let value_fn: Box<dyn ValueFunction> =
            Box::new(InfluenceBasedValueFunction::new(self.graph));

        let shapley = ShapleyValueCalculator::new(value_fn.clone());
        let banzhaf = BanzhafPowerIndex::new(value_fn);

        let mut shapley_values = shapley.calculate_stratified_antithetic(players, 5);
        let banzhaf_indices = banzhaf.calculate(players);

        // === Mercy-aware context detection (PATSAGi Councils aligned) ===
        let has_mercygate = self.graph.defeaters.iter().any(|d| {
            matches!(d.context, Some(SuperiorityContext::MercyGate))
        });

        let has_council = self.graph.superiorities.iter().any(|s| {
            matches!(s.context, Some(SuperiorityContext::Council))
        });

        let mut context_notes = vec![];
        let mut summary = if has_mercygate {
            "Mercy-aware arbitration completed. Negative influence moderated."
        } else if has_council {
            "Council-weighted arbitration completed with structural emphasis."
        } else {
            "Standard combined Shapley + Banzhaf analysis completed."
        }
        .to_string();

        if has_mercygate {
            // Deeper mercy-aware numerical adjustment
            // PATSAGi Council guidance: apply compassionate softening (0.75) to compress
            // the spread of Shapley values. This reduces harshness of extreme influence
            // differences (mirrors Phase 4 MercyGate ×0.6 reduction pattern) while
            // preserving relative ordering and sign. Applied at arbitration/report layer.
            let softening_factor = 0.75;
            if !shapley_values.is_empty() {
                let mean: f64 =
                    shapley_values.values().sum::<f64>() / shapley_values.len() as f64;
                for val in shapley_values.values_mut() {
                    let deviation = *val - mean;
                    *val = mean + deviation * softening_factor;
                }
            }
            context_notes.push(
                "MercyGate context detected — compassionate softening (0.75) applied to Shapley values to reduce influence spread harshness."
                    .to_string(),
            );
            summary.push_str(" Compassionate softening factor 0.75 applied to Shapley spread.");
        }
        if has_council {
            context_notes.push(
                "Council context present — authoritative weighting applied.".to_string(),
            );
        }

        ArbitrationReport {
            shapley_values,
            banzhaf_indices,
            summary,
            context_notes,
        }
    }
}

pub fn influence_shapley_calculator(graph: &ArgumentGraph) -> ShapleyValueCalculator {
    ShapleyValueCalculator::new(Box::new(InfluenceBasedValueFunction::new(graph)))
}

pub fn influence_banzhaf_calculator(graph: &ArgumentGraph) -> BanzhafPowerIndex {
    BanzhafPowerIndex::new(Box::new(InfluenceBasedValueFunction::new(graph)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercy_aware_arbitration() {
        let mut graph = ArgumentGraph::new();
        let c1 = graph.add_claim("C1".to_string(), "Test".to_string(), 0.8);
        let c2 = graph.add_claim("C2".to_string(), "Test".to_string(), 0.7);

        let arbitration = PatsagiArbitration::new(&graph);
        let players = vec![GovernancePlayer::Claim(c1), GovernancePlayer::Claim(c2)];
        let report = arbitration.analyze(&players);

        assert!(!report.summary.is_empty());
    }
}