//! crates/powrush/src/simulation.rs
//! High-level WorldSimulation / Game Loop for Powrush
//! Wires NpcIntegration + Geometric Harmony + RBE Economy Credits | v15 Hybrid + ONE Organism
//! AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing};
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

/// High-level world simulation with v15 NPC, Geometric Harmony scoring,
/// and RBE economy credit from epigenetic blessings.
pub struct WorldSimulation {
    pub npc_integration: NpcIntegration,
    pub player: PlayerState,
    pub world_mercy: f64,
    pub is_post_scarcity: bool,
    pub collective_joy: f64,
    pub tick_count: u64,

    // === New: Geometric + Economy ===
    pub geometric_harmony_score: f64,
    pub economy_credits: f64,
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
            economy_credits: 0.0,
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

        // === 1. Geometric Harmony Scoring (wired for geometric-intelligence crate) ===
        self.geometric_harmony_score = self.compute_geometric_harmony();

        // === 2. RBE Economy Credits from Epigenetic Blessings ===
        let total_blessing = self.distribute_blessings_to_economy();
        self.economy_credits += total_blessing;
    }

    /// Placeholder that will call geometric-intelligence::compute_geometric_harmony
    /// once the full engine is integrated. Currently uses spatial spread as proxy.
    fn compute_geometric_harmony(&self) -> f64 {
        if self.npc_integration.npc_system.agents.is_empty() {
            return 0.75;
        }
        // Simple proxy: average distance from center + mercy influence
        let center = Vector2::new(0.0, 0.0);
        let avg_dist: f64 = self.npc_integration.npc_system.agents
            .iter()
            .map(|a| (a.position - center).magnitude() as f64)
            .sum::<f64>() / self.npc_integration.npc_system.agents.len() as f64;

        let mercy_factor = self.npc_integration.npc_system.agents
            .iter()
            .map(|a| a.blackboard.current_mercy_valence)
            .sum::<f64>() / self.npc_integration.npc_system.agents.len() as f64;

        (0.6 + (avg_dist.min(25.0) / 25.0) * 0.2 + mercy_factor * 0.2).min(0.99)
    }

    /// Distributes epigenetic blessings and returns total credits added to economy
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
}