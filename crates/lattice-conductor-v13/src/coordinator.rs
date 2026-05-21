//! Multi-conductor coordination and pluggable strategies for v13.
//! Layered coordination per LAYERED_COORDINATION_ARCHITECTURE.md

use crate::{SimpleLatticeConductor, MercyWeightedVote};
use std::sync::Arc;

/// Strategy trait for influencing multiple conductors (pluggable at runtime).
pub trait CoordinationStrategy: Send + Sync {
    fn influence(&self, conductors: &mut [SimpleLatticeConductor]);
    fn name(&self) -> &'static str;
}

/// Mercy-weighted group influence (recommended default for Layer 1/2).
pub struct MercyWeightedStrategy;

impl CoordinationStrategy for MercyWeightedStrategy {
    fn influence(&self, conductors: &mut [SimpleLatticeConductor]) {
        if conductors.is_empty() { return; }
        let mut group_vote = MercyWeightedVote::new();
        for c in conductors.iter() {
            group_vote.add_vote(&c.name, 1.0 / conductors.len() as f64, c.state.mercy_score - 1.0);
        }
        let consensus = group_vote.compute_consensus();
        for c in conductors.iter_mut() {
            c.state.mercy_score = (c.state.mercy_score + consensus * 0.2).clamp(0.3, 1.5);
            c.adaptive_params.mercy_recovery_rate *= 1.01; // evolve
        }
    }

    fn name(&self) -> &'static str { "MercyWeightedStrategy" }
}

/// Leader-follower influence.
pub struct LeaderFollowerStrategy;

impl CoordinationStrategy for LeaderFollowerStrategy {
    fn influence(&self, conductors: &mut [SimpleLatticeConductor]) {
        if conductors.len() < 2 { return; }
        let leader_mercy = conductors[0].state.mercy_score;
        for c in conductors.iter_mut().skip(1) {
            c.state.mercy_score = (c.state.mercy_score * 0.7 + leader_mercy * 0.3).clamp(0.3, 1.5);
            c.adaptive_params.evolution_rate += 0.001;
        }
    }

    fn name(&self) -> &'static str { "LeaderFollowerStrategy" }
}

/// Simple average influence.
pub struct AverageInfluenceStrategy;

impl CoordinationStrategy for AverageInfluenceStrategy {
    fn influence(&self, conductors: &mut [SimpleLatticeConductor]) {
        if conductors.is_empty() { return; }
        let avg_mercy: f64 = conductors.iter().map(|c| c.state.mercy_score).sum::<f64>() / conductors.len() as f64;
        for c in conductors.iter_mut() {
            c.state.mercy_score = (c.state.mercy_score + (avg_mercy - c.state.mercy_score) * 0.3).clamp(0.3, 1.5);
        }
    }

    fn name(&self) -> &'static str { "AverageInfluenceStrategy" }
}

/// Multi-conductor simulation engine.
/// Supports dynamic strategy switching and coordinated ticks.
pub struct MultiConductorSimulation {
    pub conductors: Vec<SimpleLatticeConductor>,
    strategy: Box<dyn CoordinationStrategy>,
}

impl MultiConductorSimulation {
    pub fn with_strategy(strategy: Box<dyn CoordinationStrategy>) -> Self {
        Self {
            conductors: Vec::new(),
            strategy,
        }
    }

    pub fn add_conductor(&mut self, conductor: SimpleLatticeConductor) {
        self.conductors.push(conductor);
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn CoordinationStrategy>) {
        self.strategy = strategy;
    }

    /// Execute one coordinated step across all conductors.
    pub fn coordinated_tick(&mut self) {
        self.strategy.influence(&mut self.conductors);
        for c in &mut self.conductors {
            let _ = c.tick();
        }
    }
}
