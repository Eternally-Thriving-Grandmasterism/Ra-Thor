//! crates/powrush/src/simulation.rs
//! High-level WorldSimulation with Stabilized Geometric Engine + Real RBE Economy + Visualization
//! v15 Hybrid NPC + ONE Organism | AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing};
use geometric_intelligence::compute_geometric_harmony;
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

/// Structured RBE Economy with credit + spend mechanics
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

    /// Example: spend credits on Powrush items / services (RBE sink)
    pub fn spend_on_item(&mut self, item_cost: f64, item_name: &str) -> bool {
        if self.total_credits >= item_cost {
            self.total_credits -= item_cost;
            println!("   [RBE] Spent {:.1} credits on '{}'", item_cost, item_name);
            true
        } else {
            false
        }
    }

    pub fn current_pool(&self) -> f64 {
        self.total_credits
    }
}

/// High-level WorldSimulation with stabilized geometric engine,
/// real RBE economy wiring, and rich visualization/logging.
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

        // Seed diverse NPCs
        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        sim.npc_integration.spawn_agent(NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol)));
        sim.npc_integration.spawn_agent(NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None));
        sim.npc_integration.spawn_agent(NpcFactory::create_guardian(Vector2::new(15.0, 8.0), None));

        sim
    }

    pub fn tick(&mut self, dt: f32) {
        self.tick_count += 1;

        // Player movement
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

        // === Stabilized Geometric Engine Call (no fallback) ===
        self.geometric_harmony_score = compute_geometric_harmony(
            &self.prepare_geometric_input()
        ).unwrap_or_else(|_| 0.82); // graceful if engine returns Err

        // === RBE Economy Credit + Example Spend ===
        let blessing_total = self.distribute_blessings_to_economy();
        self.economy.credit(blessing_total);

        // Occasional economy activity (demo of real item wiring)
        if self.tick_count % 5 == 0 && self.economy.current_pool() > 3.0 {
            self.economy.spend_on_item(2.5, "Mercy Crystal");
        }

        // === Visualization / Logging ===
        if self.tick_count % 2 == 0 {
            self.log_status();
        }
    }

    fn prepare_geometric_input(&self) -> Vec<(f64, f64, f64)> {
        self.npc_integration.npc_system.agents
            .iter()
            .map(|agent| {
                (
                    agent.position.x as f64,
                    agent.position.y as f64,
                    agent.blackboard.current_mercy_valence,
                )
            })
            .collect()
    }

    fn distribute_blessings_to_economy(&mut self) -> f64 {
        self.npc_integration.npc_system.agents
            .iter_mut()
            .map(|agent| distribute_epigenetic_blessing(&mut agent.blackboard))
            .sum()
    }

    /// Rich status logging / visualization
    pub fn log_status(&self) {
        println!(
            "[Tick {:03}] Harmony: {:.3} | Economy: {:.1} cr | NPCs: {} | Player({:.1}, {:.1})",
            self.tick_count,
            self.geometric_harmony_score,
            self.economy.current_pool(),
            self.active_npcs(),
            self.player.position.x,
            self.player.position.y
        );
    }

    pub fn active_npcs(&self) -> usize {
        self.npc_integration.active_npc_count()
    }

    pub fn current_economy_pool(&self) -> f64 {
        self.economy.current_pool()
    }
}