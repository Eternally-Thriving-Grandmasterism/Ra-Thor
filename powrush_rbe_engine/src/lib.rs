/*!
# Powrush RBE Engine — Core Economy Rules

Professional implementation of Resource-Based Economy rules for the Powrush MMO.

## Core Principles (Encoded)
- **Abundance over Scarcity**: Resources are fundamentally abundant. Artificial scarcity is rejected.
- **Contribution + Mercy Floor**: Access and distribution weighted by contribution to the commons, with a guaranteed mercy floor for universal thriving.
- **Production = Capacity × Harmony × Technology**: No arbitrary limits; output scales with real factors and council-modulated harmony.
- **Economic Proposals**: Major distribution or production policy changes route through PATSAGi Council proposals for mercy-gated approval.
- **Epigenetic & Geometric Modulation**: Economic outcomes influenced by shard harmony, epigenetic blessings, and sacred geometry layers.

This crate provides pure, testable rules that `powrush-mmo-simulator` (and future client/server layers) can call every tick.

All code follows the Eternal Iteration Protocol and AG-SML v1.0.
*/

use geometric_intelligence::{ShardManager, EpigeneticBlessing};
use std::collections::HashMap;

/// Core resource types in Powrush RBE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Resource {
    Energy,
    Materials,
    Knowledge,
    BioMass,
    Data,
    QuantumFlux,
}

/// Output from a production cycle
#[derive(Debug, Clone)]
pub struct ProductionOutput {
    pub resource: Resource,
    pub amount: f64,
    pub efficiency: f64, // 0.0 - 1.0+
    pub harmony_bonus: f64,
}

/// Player or faction contribution to the commons
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

/// Result of a distribution round
#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub allocations: HashMap<String, f64>, // id -> amount received
    pub mercy_floor_applied: f64,
    pub total_distributed: f64,
}

/// Main RBE Economy state and rule engine
#[derive(Debug, Clone)]
pub struct RBEconomy {
    pub abundance_index: f64,           // Global 0.0 - ~2.0+
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

    /// Core production rule: Capacity × Harmony × TechLevel (with mercy modulation)
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

        // Apply epigenetic blessings (Abundance & Service gates)
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

    /// Contribution-weighted distribution with mercy floor for universal thriving
    /// (Classic RBE + Mercy overlay)
    pub fn distribute(
        &self,
        total_available: f64,
        contributions: &[Contribution],
        mercy_floor: f64, // minimum per participant (0.0 - 0.3 typical)
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
            // Equal distribution if no contributions recorded
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

    /// Update global and shard abundance based on production vs consumption
    pub fn update_abundance(&mut self, production: f64, consumption: f64, shard_id: Option<&str>) {
        let net = production - consumption;
        self.abundance_index = (self.abundance_index + net * 0.01).clamp(0.4, 2.5);

        if let Some(sid) = shard_id {
            let current = self.shard_abundances.get(sid).copied().unwrap_or(1.0);
            let new_val = (current + net * 0.015).clamp(0.5, 2.8);
            self.shard_abundances.insert(sid.to_string(), new_val);
        }
    }

    /// Record a contribution (called from simulator or player actions)
    pub fn record_contribution(&mut self, contrib: Contribution) {
        self.contributions.insert(contrib.id.clone(), contrib);
    }

    /// Apply council-blessed economic modulation (from ShardManager routed proposals)
    pub fn apply_council_modulation(&mut self, blessings: &[EpigeneticBlessing]) {
        for b in blessings {
            // Truth + Abundance gates: boost overall abundance
            self.abundance_index = (self.abundance_index + b.magnitude * 0.03).clamp(0.4, 2.8);
            // Service gate: slightly increase mercy floor effect in future distributions
        }
    }

    /// Full economy tick step (can be called from powrush-mmo-simulator)
    pub fn economy_tick(
        &mut self,
        shard_manager: &mut ShardManager,
        base_production_capacity: f64,
        current_harmony: f64,
        tech_level: f64,
    ) -> (f64, DistributionResult) {
        // 1. Calculate production across key resources
        let energy_out = self.calculate_production(Resource::Energy, base_production_capacity, current_harmony, tech_level, &[]);
        let materials_out = self.calculate_production(Resource::Materials, base_production_capacity * 0.8, current_harmony, tech_level, &[]);

        let total_produced = energy_out.amount + materials_out.amount;

        // 2. Simple consumption model (scales with abundance and activity)
        let consumption = self.abundance_index * 0.6 + 0.3;

        // 3. Update abundance
        self.update_abundance(total_produced, consumption, Some("hyperbolic_core"));

        // 4. Prepare contributions for distribution (placeholder — real data comes from simulator)
        let contribs: Vec<Contribution> = self.contributions.values().cloned().collect();
        let distribution = self.distribute(total_produced * 0.7, &contribs, 0.15); // 15% mercy floor

        // 5. Record last production
        self.last_tick_production.insert(Resource::Energy, energy_out.amount);
        self.last_tick_production.insert(Resource::Materials, materials_out.amount);

        (total_produced, distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_and_distribution() {
        let economy = RBEconomy::new();
        let prod = economy.calculate_production(Resource::Energy, 100.0, 0.95, 1.2, &[]);
        assert!(prod.amount > 100.0);

        let contribs = vec![
            Contribution { id: "player1".into(), amount: 50.0, kind: ContributionKind::Production },
            Contribution { id: "player2".into(), amount: 30.0, kind: ContributionKind::Innovation },
        ];
        let dist = economy.distribute(200.0, &contribs, 0.2);
        assert!(dist.mercy_floor_applied > 0.0);
        assert!(dist.allocations.len() == 2);
    }

    #[test]
    fn test_economy_tick() {
        let mut economy = RBEconomy::new();
        let mut sm = ShardManager::new();
        sm.create_shard("hyperbolic_core", "evolutionary");

        let (produced, dist) = economy.economy_tick(&mut sm, 120.0, 0.92, 1.1);
        assert!(produced > 0.0);
        assert!(economy.abundance_index > 0.9);
    }
}
