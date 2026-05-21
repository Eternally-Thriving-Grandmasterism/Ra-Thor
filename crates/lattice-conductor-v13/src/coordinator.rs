/// Multi-Agent Coordination Module
/// Phase 2: Pluggable Coordination Strategies

use crate::{ConductorEvent, ConductorResult, SimpleLatticeConductor};
use std::collections::VecDeque;

// ============================================================================
// COORDINATION STRATEGY TRAIT
// ============================================================================

/// Strategy that defines how multiple conductors coordinate with each other
pub trait CoordinationStrategy {
    /// Apply coordination influence across the group of conductors
    fn apply(&self, conductors: &mut [SimpleLatticeConductor], strength: f64);

    /// Name of the strategy (for debugging/logging)
    fn name(&self) -> &'static str;
}

// ============================================================================
// CONCRETE STRATEGIES
// ============================================================================

/// Simple averaging strategy - conductors gently pull toward group average
pub struct AverageInfluenceStrategy;

impl CoordinationStrategy for AverageInfluenceStrategy {
    fn apply(&self, conductors: &mut [SimpleLatticeConductor], strength: f64) {
        if conductors.len() <= 1 || strength < 0.05 {
            return;
        }

        let avg_evolution: f64 =
            conductors.iter().map(|c| c.state.evolution_level).sum::<f64>() / conductors.len() as f64;

        for conductor in conductors.iter_mut() {
            let diff = avg_evolution - conductor.state.evolution_level;
            conductor.state.evolution_level += diff * strength * 0.1;
        }
    }

    fn name(&self) -> &'static str {
        "AverageInfluence"
    }
}

/// Mercy-weighted strategy - conductors with higher mercy have more influence
pub struct MercyWeightedStrategy;

impl CoordinationStrategy for MercyWeightedStrategy {
    fn apply(&self, conductors: &mut [SimpleLatticeConductor], strength: f64) {
        if conductors.len() <= 1 || strength < 0.05 {
            return;
        }

        let total_mercy: f64 = conductors.iter().map(|c| c.state.mercy_score).sum();
        if total_mercy < 0.1 {
            return;
        }

        let weighted_avg: f64 = conductors
            .iter()
            .map(|c| c.state.evolution_level * c.state.mercy_score)
            .sum::<f64>() / total_mercy;

        for conductor in conductors.iter_mut() {
            let diff = weighted_avg - conductor.state.evolution_level;
            let resistance = 0.3 + (conductor.state.mercy_score * 0.5);
            conductor.state.evolution_level += diff * strength * resistance;
        }
    }

    fn name(&self) -> &'static str {
        "MercyWeighted"
    }
}

/// Leader-Follower strategy (first conductor acts as leader)
pub struct LeaderFollowerStrategy;

impl CoordinationStrategy for LeaderFollowerStrategy {
    fn apply(&self, conductors: &mut [SimpleLatticeConductor], strength: f64) {
        if conductors.len() <= 1 || strength < 0.05 {
            return;
        }

        let leader_evolution = conductors[0].state.evolution_level;

        for conductor in conductors.iter_mut().skip(1) {
            let diff = leader_evolution - conductor.state.evolution_level;
            conductor.state.evolution_level += diff * strength * 0.15;
        }
    }

    fn name(&self) -> &'static str {
        "LeaderFollower"
    }
}

// ============================================================================
// MULTI CONDUCTOR SIMULATION (with pluggable strategies)
// ============================================================================

pub trait ConductorCoordinator {
    fn coordinate_step(&mut self) -> ConductorResult<()>;
    fn conductor_count(&self) -> usize;
    fn set_coordination_strength(&mut self, strength: f64);
}

pub struct MultiConductorSimulation {
    pub conductors: Vec<SimpleLatticeConductor>,
    pub coordination_strength: f64,
    pub strategy: Box<dyn CoordinationStrategy>,
    pub coordination_events: VecDeque<ConductorEvent>,
}

impl MultiConductorSimulation {
    pub fn new() -> Self {
        Self {
            conductors: Vec::new(),
            coordination_strength: 0.25,
            strategy: Box::new(AverageInfluenceStrategy),
            coordination_events: VecDeque::with_capacity(128),
        }
    }

    pub fn with_strategy(strategy: Box<dyn CoordinationStrategy>) -> Self {
        Self {
            conductors: Vec::new(),
            coordination_strength: 0.25,
            strategy,
            coordination_events: VecDeque::with_capacity(128),
        }
    }

    pub fn add_conductor(&mut self, conductor: SimpleLatticeConductor) {
        self.conductors.push(conductor);
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn CoordinationStrategy>) {
        self.strategy = strategy;
    }

    pub fn coordinated_tick(&mut self) -> ConductorResult<()> {
        if self.conductors.is_empty() {
            return Ok(());
        }

        for conductor in &mut self.conductors {
            let _ = conductor.tick();
        }

        let avg_mercy: f64 = self.conductors
            .iter()
            .map(|c| c.state.mercy_score)
            .sum::<f64>() / self.conductors.len() as f64;

        if avg_mercy > 0.6 && self.coordination_strength > 0.05 {
            self.strategy.apply(&mut self.conductors, self.coordination_strength);

            self.coordination_events.push_back(ConductorEvent::SwarmEvolved {
                branches: self.conductors.len() as u32,
                coherence: self.coordination_strength,
            });

            if self.coordination_events.len() > 64 {
                self.coordination_events.pop_front();
            }
        }

        Ok(())
    }

    pub fn set_coordination_strength(&mut self, strength: f64) {
        self.coordination_strength = strength.clamp(0.0, 1.0);
    }
}

impl ConductorCoordinator for MultiConductorSimulation {
    fn coordinate_step(&mut self) -> ConductorResult<()> {
        self.coordinated_tick()
    }

    fn conductor_count(&self) -> usize {
        self.conductors.len()
    }

    fn set_coordination_strength(&mut self, strength: f64) {
        self.set_coordination_strength(strength);
    }
}