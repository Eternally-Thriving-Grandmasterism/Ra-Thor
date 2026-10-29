//! crates/powrush/src/npc/system.rs
//! High-level NPC simulation system — wires the full v15 hybrid NPC AI into the game loop
//! Mercy, world state, and post-scarcity as first-class inputs | AG-SML v1.0

use super::NpcAgent;

/// Manages all NPCs in the world and runs the full hybrid decision loop each tick.
pub struct NpcSystem {
    pub agents: Vec<NpcAgent>,
}

impl NpcSystem {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Spawn a new NPC into the simulation.
    pub fn spawn(&mut self, agent: NpcAgent) {
        self.agents.push(agent);
    }

    /// Main simulation tick — call this once per game frame from `game.rs`.
    ///
    /// This runs the complete v15 hybrid pipeline for every NPC:
    /// Perception → Blackboard sync → Patrol → Consideration scoring → UtilityAction selection
    pub fn update(
        &mut self,
        world_mercy: f64,
        is_post_scarcity: bool,
        collective_joy: f64,
        delta_time: f32,
    ) {
        for agent in &mut self.agents {
            // Inject current world state (mercy as first-class data)
            agent.blackboard.update_world_state(world_mercy, is_post_scarcity);
            agent.blackboard.collective_joy = collective_joy;

            // Run one full hybrid decision cycle
            agent.tick(delta_time);
        }
    }

    pub fn active_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for NpcSystem {
    fn default() -> Self {
        Self::new()
    }
}