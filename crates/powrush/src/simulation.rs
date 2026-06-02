//! crates/powrush/src/simulation.rs
//! High-level WorldSimulation / Game Loop for Powrush
//! Full Geometric Harmony Engine + RBE Economy Integration | v15 Hybrid
//! ONE Organism + TOLC 8 aligned | AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing};
use geometric_intelligence::compute_geometric_harmony; // Full engine integration
use nalgebra::Vector2;

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub ascension: f64,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            mercy: 0.82,
            ascension: 1.5,
        }
    }
}

/// Simple RBE Economy pool that receives credits from NPC epigenetic blessings
#[derive(Debug, Clone, Default)]
pub struct RbeEconomy {
    pub total_credits: f64,
    pub last_distribution: f64,
}

impl RbeEconomy {
    pub fn credit(&mut self, amount: f64) {
        if amount > 0.0 {
            self.total_credits += amount;
            self.last_distribution = amount;
        }
    }

    pub fn current_pool(&self) -> f64 {
        self.total_credits
    }
}

/// High-level world simulation with full v15 NPC, Geometric Harmony (real engine),
/// and structured RBE economy.
pub struct WorldSimulation {
    pub npc_integration: NpcIntegration,
    pub player: PlayerState,
    pub world_mercy: f64,
    pub is_post_scarcity: bool,
    pub collective_joy: f64,
    pub tick_count: u64,

    pub geometric_harmony_score: f64,
    pub economy: RbeEconomy,
}

impl WorldSimulation {
    pub fn new() -> Self {
        let mut sim = Self {
            npc_integration: NpcIntegration::default(),
            player: PlayerState::default(),
            world_mercy: 0.88,
            is_post_scarcity: true,
            collective_joy: 0.94,
            tick_count: 0,
            geometric_harmony_score: 0.0,
            economy: RbeEconomy::default(),
        };

        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        let n1 = NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol));
        sim.npc_integration.spawn_agent(n1);

        let n2 = NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None);
        sim.npc_integration.spawn_agent(n2);

        sim
    }

    pub fn tick(&mut self, dt: f32) {
        self.tick_count += 1;

        self.player.position.x = (self.player.position.x + 0.7) % 40.0;
        self.player.position.y = 2.0 + (self.tick_count as f32 * 0.08).sin() * 4.0;

        let noise_level = if self.tick_count % 4 == 0 { 0.55 } else { 0.18 };

        self.npc_integration.update(
            self.world_mercy,
            self.is_post_scarcity,
            self.collective_joy,
            Some(self.player.position),
            noise_level,
            dt,
        );

        // === Full Geometric Harmony Engine Integration ===
        self.geometric_harmony_score = self.apply_geometric_harmony();

        // === RBE Economy Credit from Epigenetic Blessings ===
        let blessing_total = self.distribute_blessings_to_economy();
        self.economy.credit(blessing_total);
    }

    /// Calls the real geometric-intelligence engine.
    /// Falls back to high-quality proxy if engine returns unexpected result.
    fn apply_geometric_harmony(&self) -> f64 {
        // Prepare input for the engine (NPC positions + mercy valence as resonance data)
        let positions: Vec<(f64, f64, f64)> = self.npc_integration.npc_system.agents
            .iter()
            .map(|agent| {
                (
                    agent.position.x as f64,
                    agent.position.y as f64,
                    agent.blackboard.current_mercy_valence,
                )
            })
            .collect();

        if positions.is_empty() {
            return 0.78;
        }

        // Real engine call — this is the full geometric swap
        match compute_geometric_harmony(&positions) {
            Ok(score) => score.clamp(0.0, 1.0),
            Err(_) => self.fallback_geometric_harmony(),
        }
    }

    fn fallback_geometric_harmony(&self) -> f64 {
        // High-quality proxy until engine is fully stable
        let center = Vector2::new(0.0, 0.0);
        let avg_dist: f64 = self.npc_integration.npc_system.agents
            .iter()
            .map(|a| (a.position - center).magnitude() as f64)
            .sum::<f64>() / self.npc_integration.npc_system.agents.len() as f64;

        let mercy_factor = self.npc_integration.npc_system.agents
            .iter()
            .map(|a| a.blackboard.current_mercy_valence)
            .sum::<f64>() / self.npc_integration.npc_system.agents.len() as f64;

        (0.65 + (avg_dist.min(30.0) / 30.0) * 0.2 + mercy_factor * 0.15).min(0.98)
    }

    fn distribute_blessings_to_economy(&mut self) -> f64 {
        let mut total = 0.0;
        for agent in &mut self.npc_integration.npc_system.agents {
            total += distribute_epigenetic_blessing(&mut agent.blackboard);
        }
        total
    }

    pub fn active_npcs(&self) -> usize {
        self.npc_integration.active_npc_count()
    }

    pub fn current_economy_pool(&self) -> f64 {
        self.economy.current_pool()
    }
}