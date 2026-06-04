/*!
# Powrush RBE Engine — Core Economy Rules (Enhanced)

Professional RBE rules with support for dynamic council-modulated mercy floor.

Changes in this iteration:
- `economy_tick` now accepts `mercy_floor: f64` parameter for dynamic council influence.
- Thoughtful design: Higher mercy_floor (from harmony/abundance council approvals) leads to more universal distribution.
*/

use geometric_intelligence::EpigeneticBlessing;
use std::collections::HashMap;

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
}

#[derive(Debug, Clone)]
pub struct RBEconomy {
    pub abundance_index: f64,
    pub shard_abundances: HashMap<String, f64>,
    pub contributions: HashMap<String, Contribution>,
    pub last_tick_production: HashMap<Resource, f64>,
}

impl RBEconomy {
    pub fn new() -> Self {
        let mut shard_abundances = HashMap::new();
        shard_abundances.insert("hyperbolic_core".to_string(), 1.0);
        shard_abundances.insert("forge_shard".to_string(), 0.9);
        shard_abundances.insert("platonic_harmony".to_string(), 1.1);

        Self {
            abundance_index: 1.0,
            shard_abundances,
            contributions: HashMap::new(),
            last_tick_production: HashMap::new(),
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
    ) -> DistributionResult {
        if contributions.is_empty() || total_available <= 0.0 {
            return DistributionResult {
                allocations: HashMap::new(),
                mercy_floor_applied: 0.0,
                total_distributed: 0.0,
            };
        }

        let total_contrib: f64 = contributions.iter().map(|c| c.amount).sum();
        if total_contrib <= 0.0 {
            let equal = total_available / contributions.len() as f64;
            let mut allocations = HashMap::new();
            for c in contributions {
                allocations.insert(c.id.clone(), equal);
            }
            return DistributionResult { allocations, mercy_floor_applied: equal, total_distributed: total_available };
        }

        let mut allocations = HashMap::new();
        let mut distributed = 0.0;
        let floor_total = mercy_floor * contributions.len() as f64;
        let remaining = (total_available - floor_total).max(0.0);

        for contrib in contributions {
            let share = if total_contrib > 0.0 {
                (contrib.amount / total_contrib) * remaining
            } else {
                0.0
            };
            let received = mercy_floor + share;
            allocations.insert(contrib.id.clone(), received);
            distributed += received;
        }

        DistributionResult {
            allocations,
            mercy_floor_applied: mercy_floor,
            total_distributed: distributed.min(total_available),
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

    /// economy_tick now accepts dynamic mercy_floor from council modulation
    pub fn economy_tick(
        &mut self,
        shard_manager: &mut geometric_intelligence::ShardManager,
        base_production_capacity: f64,
        current_harmony: f64,
        tech_level: f64,
        mercy_floor: f64,
    ) -> (f64, DistributionResult) {
        let energy_out = self.calculate_production(Resource::Energy, base_production_capacity, current_harmony, tech_level, &[]);
        let materials_out = self.calculate_production(Resource::Materials, base_production_capacity * 0.8, current_harmony, tech_level, &[]);

        let total_produced = energy_out.amount + materials_out.amount;
        let consumption = self.abundance_index * 0.6 + 0.3;

        self.update_abundance(total_produced, consumption, Some("hyperbolic_core"));

        let contribs: Vec<Contribution> = self.contributions.values().cloned().collect();
        let distribution = self.distribute(total_produced * 0.7, &contribs, mercy_floor);

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

        let (produced, dist) = economy.economy_tick(&mut sm, 120.0, 0.92, 1.1, 0.25); // higher mercy floor
        assert!(dist.mercy_floor_applied > 0.2);
    }
}
