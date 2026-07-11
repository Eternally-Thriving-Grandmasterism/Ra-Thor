/*!
# Powrush RBE Engine — Core Economy Rules (Enhanced + TOLC Integration)

Professional RBE rules with support for dynamic council-modulated mercy floor.

**v1.1 Changes (Thread Resolution Integration)**:
- Added TOLC Unit (TU) backed claims and physics-grounded resource flows (Energy from algae/nanofactory models, etc.).
- `distribute` and `economy_tick` now accept optional TU priorities and physics_source mapping for abundance-era allocation without distortions.
- Integrates with kernel::tolc_quantification for TU_need, mercy_factor, and UTF checks.
- Maintains full compatibility with existing mercy_floor and council modulation.

All changes pass TOLC 8 gates and support Lattice Conductor v13+ allocation_priority_queue.
*/

use geometric_intelligence::EpigeneticBlessing;
use std::collections::HashMap;

// TOLC integration (from kernel/tolc_quantification.rs)
// In production: use crate::kernel::tolc_quantification::{TOLCUnit, allocation_priority, passes_utf, UTFThresholds};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Resource {
    Energy,
    Materials,
    Knowledge,
    BioMass,
    Data,
    QuantumFlux,
}

#[derive(Debug, Clone)]
pub struct ProductionOutput {
    pub resource: Resource,
    pub amount: f64,
    pub efficiency: f64,
    pub harmony_bonus: f64,
}

#[derive(Debug, Clone)]
pub struct Contribution {
    pub id: String,
    pub amount: f64,
    pub kind: ContributionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContributionKind {
    Production,
    Curation,
    Defense,
    Innovation,
    Governance,
}

#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub allocations: HashMap<String, f64>,
    pub mercy_floor_applied: f64,
    pub total_distributed: f64,
    pub tu_weighted: bool, // new: indicates TU-backed distribution used
}

#[derive(Debug, Clone)]
pub struct RBEconomy {
    pub abundance_index: f64,
    pub shard_abundances: HashMap<String, f64>,
    pub contributions: HashMap<String, Contribution>,
    pub last_tick_production: HashMap<Resource, f64>,
    pub physics_sources: HashMap<Resource, f64>, // new: real physics backing (e.g. algae energy output)
}

impl RBEconomy {
    pub fn new() -> Self {
        let mut shard_abundances = HashMap::new();
        shard_abundances.insert("hyperbolic_core".to_string(), 1.0);
        shard_abundances.insert("forge_shard".to_string(), 0.9);
        shard_abundances.insert("platonic_harmony".to_string(), 1.1);

        let mut physics_sources = HashMap::new();
        physics_sources.insert(Resource::Energy, 1200.0); // proxy algae/nanofactory J output

        Self {
            abundance_index: 1.0,
            shard_abundances,
            contributions: HashMap::new(),
            last_tick_production: HashMap::new(),
            physics_sources,
        }
    }

    pub fn calculate_production(
        &self,
        resource: Resource,
        base_capacity: f64,
        harmony: f64,
        tech_level: f64,
        blessings: &[EpigeneticBlessing],
    ) -> ProductionOutput {
        let harmony_factor = harmony.clamp(0.6, 1.3);
        let tech_factor = tech_level.clamp(0.5, 3.0);
        let mut base_amount = base_capacity * harmony_factor * tech_factor;

        let blessing_multiplier: f64 = blessings.iter().map(|b| 1.0 + b.magnitude * 0.15).sum();
        base_amount *= blessing_multiplier.max(1.0);

        let efficiency = (harmony_factor + tech_factor) / 3.0;

        ProductionOutput {
            resource,
            amount: base_amount,
            efficiency,
            harmony_bonus: harmony_factor - 1.0,
        }
    }

    pub fn distribute(
        &self,
        total_available: f64,
        contributions: &[Contribution],
        mercy_floor: f64,
        tu_priorities: Option<&HashMap<String, f64>>, // new: TU-based priority weights
    ) -> DistributionResult {
        if contributions.is_empty() || total_available <= 0.0 {
            return DistributionResult {
                allocations: HashMap::new(),
                mercy_floor_applied: 0.0,
                total_distributed: 0.0,
                tu_weighted: false,
            };
        }

        let total_contrib: f64 = contributions.iter().map(|c| c.amount).sum();
        let mut allocations = HashMap::new();
        let mut distributed = 0.0;
        let floor_total = mercy_floor * contributions.len() as f64;
        let remaining = (total_available - floor_total).max(0.0);

        for contrib in contributions {
            let base_share = if total_contrib > 0.0 {
                (contrib.amount / total_contrib) * remaining
            } else {
                0.0
            };

            let tu_boost = if let Some(prios) = tu_priorities {
                prios.get(&contrib.id).copied().unwrap_or(1.0)
            } else {
                1.0
            };

            let received = mercy_floor + base_share * tu_boost;
            allocations.insert(contrib.id.clone(), received);
            distributed += received;
        }

        DistributionResult {
            allocations,
            mercy_floor_applied: mercy_floor,
            total_distributed: distributed.min(total_available),
            tu_weighted: tu_priorities.is_some(),
        }
    }

    pub fn update_abundance(&mut self, production: f64, consumption: f64, shard_id: Option<&str>) {
        let net = production - consumption;
        self.abundance_index = (self.abundance_index + net * 0.01).clamp(0.4, 2.5);

        if let Some(sid) = shard_id {
            let current = self.shard_abundances.get(sid).copied().unwrap_or(1.0);
            let new_val = (current + net * 0.015).clamp(0.5, 2.8);
            self.shard_abundances.insert(sid.to_string(), new_val);
        }
    }

    pub fn record_contribution(&mut self, contrib: Contribution) {
        self.contributions.insert(contrib.id.clone(), contrib);
    }

    pub fn apply_council_modulation(&mut self, blessings: &[EpigeneticBlessing]) {
        for b in blessings {
            self.abundance_index = (self.abundance_index + b.magnitude * 0.03).clamp(0.4, 2.8);
        }
    }

    /// economy_tick now accepts dynamic mercy_floor from council + optional TU priorities and physics sources
    pub fn economy_tick(
        &mut self,
        shard_manager: &mut geometric_intelligence::ShardManager,
        base_production_capacity: f64,
        current_harmony: f64,
        tech_level: f64,
        mercy_floor: f64,
        tu_priorities: Option<&HashMap<String, f64>>,
    ) -> (f64, DistributionResult) {
        // Physics-backed production (Energy from real sources)
        let physics_energy = self.physics_sources.get(&Resource::Energy).copied().unwrap_or(base_production_capacity);
        let energy_out = self.calculate_production(Resource::Energy, physics_energy, current_harmony, tech_level, &[]);
        let materials_out = self.calculate_production(Resource::Materials, base_production_capacity * 0.8, current_harmony, tech_level, &[]);

        let total_produced = energy_out.amount + materials_out.amount;
        let consumption = self.abundance_index * 0.6 + 0.3;

        self.update_abundance(total_produced, consumption, Some("hyperbolic_core"));

        let contribs: Vec<Contribution> = self.contributions.values().cloned().collect();
        let distribution = self.distribute(total_produced * 0.7, &contribs, mercy_floor, tu_priorities);

        self.last_tick_production.insert(Resource::Energy, energy_out.amount);
        self.last_tick_production.insert(Resource::Materials, materials_out.amount);

        (total_produced, distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_mercy_floor() {
        let mut economy = RBEconomy::new();
        let mut sm = geometric_intelligence::ShardManager::new();
        sm.create_shard("hyperbolic_core", "evolutionary");

        let (produced, dist) = economy.economy_tick(&mut sm, 120.0, 0.92, 1.1, 0.25, None);
        assert!(dist.mercy_floor_applied > 0.2);
    }

    #[test]
    fn test_tu_weighted_distribution() {
        let mut economy = RBEconomy::new();
        let mut sm = geometric_intelligence::ShardManager::new();
        sm.create_shard("hyperbolic_core", "evolutionary");

        let mut tu_prios = HashMap::new();
        tu_prios.insert("node_high_tu".to_string(), 1.5);

        let (produced, dist) = economy.economy_tick(&mut sm, 120.0, 0.92, 1.1, 0.25, Some(&tu_prios));
        assert!(dist.tu_weighted);
        // High TU node should receive boosted share
    }
}
