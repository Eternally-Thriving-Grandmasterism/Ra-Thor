//! crates/powrush/src/simulation.rs
//! High-level WorldSimulation / Game Loop for Powrush
//! Wires NpcIntegration + player state + world mercy + RBE hooks | v15 Hybrid + ONE Organism
//! AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position};
use nalgebra::Vector2;

/// Simple player state for simulation
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

/// High-level world simulation that owns the full v15 NPC system
/// and provides a clean tick() for game loops or server ticks.
pub struct WorldSimulation {
    pub npc_integration: NpcIntegration,
    pub player: PlayerState,
    pub world_mercy: f64,
    pub is_post_scarcity: bool,
    pub collective_joy: f64,
    pub tick_count: u64,
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
        };

        // Seed a few NPCs on creation
        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        let n1 = NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol));
        sim.npc_integration.spawn_agent(n1);

        let n2 = NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None);
        sim.npc_integration.spawn_agent(n2);

        sim
    }

    /// Master simulation tick — call every frame or fixed timestep
    pub fn tick(&mut self, dt: f32) {
        self.tick_count += 1;

        // Gentle player movement simulation
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

        // Future: distribute epigenetic blessings here based on NPC mercy/relationship
    }

    pub fn active_npcs(&self) -> usize {
        self.npc_integration.active_npc_count()
    }
}