//! crates/powrush/src/npc/npc_integration.rs
//! High-level integration & orchestration wiring for v15 hybrid NPC AI
//! ONE Organism participant: Perception + System tick + SpatialHash + Mercy world state
//! Production ready for Powrush game loop | PATSAGi extensible | AG-SML v1.0

use super::behavior::NpcAgent;
use super::perception::PerceptionSystem;
use super::patrol::Position;
use super::system::NpcSystem;
use super::spatial_hash::SpatialHash;

pub struct NpcIntegration {
    pub npc_system: NpcSystem,
    pub spatial_hash: SpatialHash,
    perception: PerceptionSystem,
}

impl NpcIntegration {
    pub fn new(base_cell_size: f32) -> Self {
        Self {
            npc_system: NpcSystem::new(),
            spatial_hash: SpatialHash::new(base_cell_size),
            perception: PerceptionSystem::new(),
        }
    }

    /// Spawn agent into both simulation system and spatial index.
    pub fn spawn_agent(&mut self, mut agent: NpcAgent) -> usize {
        let id = self.npc_system.agents.len();
        self.spatial_hash.insert(id, agent.position, 1.5);
        self.npc_system.spawn(agent);
        id
    }

    /// Master tick — call once per frame from Powrush simulation.
    /// Order: 1. Fresh perception (sensory injection) 2. World state + decision tick 3. Spatial sync
    pub fn update(
        &mut self,
        world_mercy: f64,
        is_post_scarcity: bool,
        collective_joy: f64,
        player_position: Option<Position>,
        global_noise_level: f32,
        dt: f32,
    ) {
        // 1. Perception pass — populate blackboard with LOS, audio, last-known using current positions
        for agent in &mut self.npc_system.agents {
            self.perception.update(
                &mut agent.blackboard,
                agent.position,
                player_position,
                global_noise_level,
                dt,
            );
        }

        // 2. Inject world mercy/ post-scarcity + run full hybrid decision loop (patrol + relationship)
        self.npc_system.update(world_mercy, is_post_scarcity, collective_joy, dt);

        // 3. Post-decision spatial hash refresh (supports future agent movement in execute_action)
        for (idx, agent) in self.npc_system.agents.iter().enumerate() {
            self.spatial_hash.insert(idx, agent.position, 1.5);
        }
    }

    pub fn get_nearby_npcs(&self, position: Position, radius: f32) -> Vec<usize> {
        self.spatial_hash.query_radius(position, radius)
    }

    pub fn active_npc_count(&self) -> usize {
        self.npc_system.active_count()
    }
}

impl Default for NpcIntegration {
    fn default() -> Self {
        Self::new(8.0)
    }
}