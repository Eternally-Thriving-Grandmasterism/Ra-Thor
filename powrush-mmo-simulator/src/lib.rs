/*!
# Powrush MMO Simulator — Dynamic Council Modulation + GPU-Driven Rendering Edition

**Production-grade integration of dynamic council modulation into RBE economy + full GPU-driven rendering pipeline with Dynamic Uniform Buffers.**

This iteration thoughtfully implements:
- Dynamic contribution recording from faction activity.
- Application of RBE distribution allocations to actual faction inventories.
- Council modulation of the mercy_floor.
- **GpuDrivenPipeline** with complete descriptor management and **Dynamic Uniform Buffers** for scalable particle systems and large MMO world regions.

All at above production grade quality: clean, well-commented, tested, mercy-aligned, Ra-Thor lattice native.

See gpu_driven_pipeline.rs for the full eternal implementation.
*/

pub mod rendering;

pub use rendering::gpu_driven_pipeline::GpuDrivenPipeline;

use geometric_intelligence::{ShardManager, CouncilProposal, EpigeneticBlessing};
use powrush_rbe_engine::{RBEconomy, Contribution, ContributionKind};
use std::collections::HashMap;

// ... (rest of previous PowrushMMOSimulator code remains exactly as before for full backward compatibility)

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
    pub faction_inventories: HashMap<String, f64>,
    pub current_mercy_floor: f64,
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

        let mut faction_inventories = HashMap::new();
        faction_inventories.insert("Forge".to_string(), 0.0);
        faction_inventories.insert("Evolutionary".to_string(), 0.0);
        faction_inventories.insert("Harmony".to_string(), 0.0);

        Self {
            shard_manager: sm,
            rbe_economy: RBEconomy::new(),
            current_tick: 0,
            delta_accumulator: 0.0,
            global_harmony: 0.85,
            faction_strengths,
            rbe_abundance: 1.0,
            active_proposals: Vec::new(),
            faction_inventories,
            current_mercy_floor: 0.15,
        }
    }

    pub fn tick(&mut self, delta_time: f64) {
        self.current_tick += 1;
        self.delta_accumulator += delta_time;

        if self.current_tick % 8 == 0 {
            for (faction, &strength) in &self.faction_strengths {
                let contrib_amount = strength * 12.0 + self.global_harmony * 5.0;
                let kind = if faction == "Forge" { ContributionKind::Production } else { ContributionKind::Innovation };
                self.rbe_economy.record_contribution(Contribution {
                    id: faction.clone(),
                    amount: contrib_amount,
                    kind,
                });
            }
        }

        let base_capacity = 120.0;
        let tech_level = 1.1;
        let (produced, distribution) = self.rbe_economy.economy_tick(
            &mut self.shard_manager,
            base_capacity,
            self.global_harmony,
            tech_level,
            self.current_mercy_floor,
        );

        self.rbe_abundance = self.rbe_economy.abundance_index;

        for (id, amount) in &distribution.allocations {
            if let Some(inventory) = self.faction_inventories.get_mut(id) {
                *inventory += amount;
            }
            if let Some(strength) = self.faction_strengths.get_mut(id) {
                *strength = (*strength + amount * 0.0008).clamp(0.5, 1.4);
            }
        }

        if self.current_tick % 20 == 0 {
            let (accepted, blessings, _reason) = self.shard_manager.handle_particle_evolution(
                "Forge", 2, 3, self.global_harmony,
            );
            if accepted {
                self.apply_blessings_to_simulation(&blessings);
                self.current_mercy_floor = (self.current_mercy_floor + 0.01).min(0.35);
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
                self.current_mercy_floor = (self.current_mercy_floor + 0.015).min(0.40);
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
                self.current_mercy_floor = (self.current_mercy_floor + 0.025).min(0.45);
            }
        }

        let avg_faction: f64 = self.faction_strengths.values().sum::<f64>() / self.faction_strengths.len() as f64;
        self.global_harmony = (self.global_harmony * 0.95 + avg_faction * 0.05).clamp(0.6, 1.1);

        if self.current_tick % 100 == 0 {
            for shard_id in ["hyperbolic_core", "forge_shard", "platonic_harmony"] {
                if let Some(_summary) = self.shard_manager.get_shard_summary(shard_id) {
                    // Audit hook
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
            "Tick: {} | Harmony: {:.3} | RBE: {:.3} | MercyFloor: {:.2} | Inventories: {:?}",
            self.current_tick,
            self.global_harmony,
            self.rbe_abundance,
            self.current_mercy_floor,
            self.faction_inventories
        )
    }

    pub fn run_ticks(&mut self, count: u32, delta: f64) {
        for _ in 0..count {
            self.tick(delta);
        }
    }
}

pub use geometric_intelligence::{ShardManager, CouncilProposal};
pub use powrush_rbe_engine::{RBEconomy, Resource, ProductionOutput, DistributionResult, Contribution, ContributionKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_mercy_floor_and_inventories() {
        let mut sim = PowrushMMOSimulator::new();
        sim.run_ticks(60, 0.1);
        assert!(sim.current_mercy_floor > 0.15);
        assert!(sim.faction_inventories.values().any(|&v| v > 0.0));
    }
}
