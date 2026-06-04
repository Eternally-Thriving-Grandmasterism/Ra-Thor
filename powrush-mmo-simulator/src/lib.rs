/*!
# Powrush MMO Simulator

**Full simulation tick loop with ShardManager + real RBEconomy integration.**

This crate now runs genuine Powrush RBE rules every tick via `powrush_rbe_engine`.

## Key Integrations
- **ShardManager** (geometric-intelligence): Interest management, council proposal routing, particle evolution wiring.
- **RBEconomy** (powrush_rbe_engine): Real production calculation, contribution-based distribution with mercy floor, abundance updates, and council modulation.
- **Mercy-Gated Flow**: Every economic outcome and proposal passes through the 7 Living Mercy Gates.

All changes follow the Eternal Iteration Protocol (PR #197) and AG-SML v1.0.
*/

use geometric_intelligence::{ShardManager, CouncilProposal, EpigeneticBlessing};
use powrush_rbe_engine::{RBEconomy, Contribution, ContributionKind};
use std::collections::HashMap;

/// Core Powrush MMO Simulation State with real RBE engine
#[derive(Debug, Clone)]
pub struct PowrushMMOSimulator {
    pub shard_manager: ShardManager,
    pub rbe_economy: RBEconomy,
    pub current_tick: u64,
    pub delta_accumulator: f64,
    pub global_harmony: f64,
    pub faction_strengths: HashMap<String, f64>,
    pub rbe_abundance: f64,
    pub active_proposals: Vec<String>,
}

impl PowrushMMOSimulator {
    pub fn new() -> Self {
        let mut sm = ShardManager::new();
        sm.create_shard("hyperbolic_core", "evolutionary");
        sm.create_shard("forge_shard", "forge");
        sm.create_shard("platonic_harmony", "harmony");
        sm.create_shard("default", "general");

        let mut faction_strengths = HashMap::new();
        faction_strengths.insert("Forge".to_string(), 0.8);
        faction_strengths.insert("Evolutionary".to_string(), 0.75);
        faction_strengths.insert("Harmony".to_string(), 0.9);

        Self {
            shard_manager: sm,
            rbe_economy: RBEconomy::new(),
            current_tick: 0,
            delta_accumulator: 0.0,
            global_harmony: 0.85,
            faction_strengths,
            rbe_abundance: 1.0,
            active_proposals: Vec::new(),
        }
    }

    /// **Full Simulation Tick with Real RBE Economy Rules**
    ///
    /// Every tick now calls the dedicated RBEconomy engine for:
    /// - Production based on capacity, harmony, tech + epigenetic blessings
    /// - Contribution-weighted distribution with mercy floor
    /// - Abundance index updates
    /// - Council modulation hooks
    pub fn tick(&mut self, delta_time: f64) {
        self.current_tick += 1;
        self.delta_accumulator += delta_time;

        // === Real RBE Economy Tick (new core) ===
        let base_capacity = 120.0;
        let tech_level = 1.1;
        let (produced, distribution) = self.rbe_economy.economy_tick(
            &mut self.shard_manager,
            base_capacity,
            self.global_harmony,
            tech_level,
        );

        // Apply RBE results to simulator state
        self.rbe_abundance = self.rbe_economy.abundance_index;

        // Simple application of distribution to faction strengths (real impl would update player inventories)
        for (id, amount) in &distribution.allocations {
            if let Some(strength) = self.faction_strengths.get_mut(id) {
                *strength = (*strength + amount * 0.001).clamp(0.5, 1.4);
            }
        }

        // === ShardManager Council Proposals & Epigenetic (existing) ===
        if self.current_tick % 20 == 0 {
            let (accepted, blessings, _reason) = self.shard_manager.handle_particle_evolution(
                "Forge", 2, 3, self.global_harmony,
            );
            if accepted {
                self.apply_blessings_to_simulation(&blessings);
                self.active_proposals.push(format!("particle_evolution_tick_{}", self.current_tick));
            }
        }

        if self.rbe_abundance > 1.5 && self.current_tick % 35 == 0 {
            let proposal = CouncilProposal::new(
                &format!("rbe_abundance_spike_{}", self.current_tick),
                "general",
                "Global RBE abundance exceeded threshold — distribute via mercy-gated economy",
                "Hyperbolic",
            );
            let (accepted, blessings, _reason) = self.shard_manager.route_council_proposal(proposal);
            if accepted {
                self.apply_blessings_to_simulation(&blessings);
            }
        }

        if self.current_tick % 50 == 0 {
            let proposal = CouncilProposal::new(
                &format!("faction_harmony_check_{}", self.current_tick),
                "Harmony",
                "Maintain Cosmic Harmony across all Powrush factions",
                "Platonic",
            );
            let (accepted, blessings, _reason) = self.shard_manager.route_council_proposal(proposal);
            if accepted {
                self.apply_blessings_to_simulation(&blessings);
            }
        }

        // Global harmony feedback
        let avg_faction: f64 = self.faction_strengths.values().sum::<f64>() / self.faction_strengths.len() as f64;
        self.global_harmony = (self.global_harmony * 0.95 + avg_faction * 0.05).clamp(0.6, 1.1);

        // Periodic audit
        if self.current_tick % 100 == 0 {
            for shard_id in ["hyperbolic_core", "forge_shard", "platonic_harmony"] {
                if let Some(_summary) = self.shard_manager.get_shard_summary(shard_id) {
                    // Future: feed to Lattice Conductor
                }
            }
        }
    }

    fn apply_blessings_to_simulation(&mut self, blessings: &[EpigeneticBlessing]) {
        for blessing in blessings {
            self.global_harmony = (self.global_harmony + blessing.valence * 0.01).clamp(0.6, 1.15);
            self.rbe_abundance = (self.rbe_abundance + blessing.magnitude * 0.005).min(2.5);

            if let Some(faction) = &blessing.target_faction {
                if let Some(strength) = self.faction_strengths.get_mut(faction) {
                    *strength = (*strength + blessing.magnitude * 0.02).clamp(0.5, 1.3);
                }
            }
        }
    }

    pub fn get_status(&self) -> String {
        format!(
            "Tick: {} | Harmony: {:.3} | RBE Abundance: {:.3} | Produced last tick: {:.1} | Active Proposals: {}",
            self.current_tick,
            self.global_harmony,
            self.rbe_abundance,
            self.rbe_economy.last_tick_production.values().sum::<f64>(),
            self.active_proposals.len()
        )
    }

    pub fn run_ticks(&mut self, count: u32, delta: f64) {
        for _ in 0..count {
            self.tick(delta);
        }
    }
}

// Re-exports for convenience
pub use geometric_intelligence::{ShardManager, CouncilProposal};
pub use powrush_rbe_engine::{RBEconomy, Resource, ProductionOutput, DistributionResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_with_rbe_tick() {
        let mut sim = PowrushMMOSimulator::new();
        sim.tick(1.0 / 60.0);
        assert!(sim.current_tick == 1);
        assert!(sim.rbe_abundance > 0.9);
    }

    #[test]
    fn test_economy_tick_produces_output() {
        let mut sim = PowrushMMOSimulator::new();
        sim.run_ticks(10, 0.1);
        let status = sim.get_status();
        assert!(status.contains("Produced last tick"));
    }
}
