/// Multi-Agent Coordination Module
/// Exploration of conductor-to-conductor coordination

use crate::{ConductorEvent, ConductorResult, SimpleLatticeConductor};
use std::collections::VecDeque;

/// Trait for any coordinator that manages multiple conductors
pub trait ConductorCoordinator {
    fn coordinate_step(&mut self) -> ConductorResult<()>;
    fn conductor_count(&self) -> usize;
    fn set_coordination_strength(&mut self, strength: f64);
}

/// Enhanced multi-conductor simulation with mercy-gated coordination
pub struct MultiConductorSimulation {
    pub conductors: Vec<SimpleLatticeConductor>,
    pub coordination_strength: f64,
    pub coordination_events: VecDeque<ConductorEvent>,
}

impl MultiConductorSimulation {
    pub fn new() -> Self {
        Self {
            conductors: Vec::new(),
            coordination_strength: 0.25,
            coordination_events: VecDeque::with_capacity(128),
        }
    }

    pub fn with_conductors(conductors: Vec<SimpleLatticeConductor>) -> Self {
        Self {
            conductors,
            coordination_strength: 0.25,
            coordination_events: VecDeque::with_capacity(128),
        }
    }

    pub fn add_conductor(&mut self, conductor: SimpleLatticeConductor) {
        self.conductors.push(conductor);
    }

    /// Coordinated tick with mercy-gated influence and event sharing
    pub fn coordinated_tick(&mut self) -> ConductorResult<()> {
        if self.conductors.is_empty() {
            return Ok(());
        }

        // Step 1: Tick all conductors independently
        for conductor in &mut self.conductors {
            let _ = conductor.tick();
        }

        // Step 2: Mercy-gated coordination influence
        // Only apply group influence if average mercy is reasonably high
        let avg_mercy: f64 = self.conductors
            .iter()
            .map(|c| c.state.mercy_score)
            .sum::<f64>() / self.conductors.len() as f64;

        if avg_mercy > 0.65 && self.coordination_strength > 0.05 && self.conductors.len() > 1 {
            // Share evolution momentum (gentle averaging)
            let avg_evolution: f64 = self.conductors
                .iter()
                .map(|c| c.state.evolution_level)
                .sum::<f64>() / self.conductors.len() as f64;

            for conductor in &mut self.conductors {
                let diff = avg_evolution - conductor.state.evolution_level;
                conductor.state.evolution_level += diff * self.coordination_strength * 0.08;
            }

            // Record coordination activity
            self.coordination_events.push_back(ConductorEvent::SwarmEvolved {
                branches: self.conductors.len() as u32,
                coherence: self.coordination_strength,
            });

            if self.coordination_events.len() > 64 {
                self.coordination_events.pop_front();
            }
        }

        // Step 3: Basic event sharing (propagate recent events to other conductors)
        // This is a simple form of multi-agent awareness
        if self.conductors.len() > 1 {
            for i in 0..self.conductors.len() {
                let recent_events = self.conductors[i].get_events();
                if !recent_events.is_empty() {
                    // For now we just record that coordination happened
                    // Future: actually inject events into other conductors
                }
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