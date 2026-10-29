//! crates/powrush/src/npc/npc_spawning.rs
//! NPC Factory for v15 Hybrid AI System
//! Templates (basic, merchant, guardian) with patrol path init + batch helpers
//! Mercy-first initialization | ONE Organism + PATSAGi ready | AG-SML v1.0

use super::behavior::NpcAgent;
use super::patrol::{PatrolManager, PatrolPath, Position};
use super::relationship::Relationship;
use nalgebra::Vector2;

pub struct NpcFactory;

impl NpcFactory {
    /// Basic patrolling NPC with optional waypoints.
    pub fn create_basic(position: Position, patrol_points: Option<Vec<Position>>) -> NpcAgent {
        let mut agent = NpcAgent::new(position);
        if let Some(points) = patrol_points {
            if !points.is_empty() {
                let path = PatrolPath::new(points);
                agent.patrol_manager = PatrolManager::with_path(path);
            }
        }
        agent.blackboard.current_mercy_valence = 0.80;
        agent.blackboard.current_behavior = "BasicPatroller".to_string();
        agent
    }

    /// Merchant NPC — high mercy bias, friendly starting relationship.
    pub fn create_merchant(position: Position, patrol_points: Option<Vec<Position>>) -> NpcAgent {
        let mut agent = Self::create_basic(position, patrol_points);
        agent.relationship = Relationship::new();
        agent.blackboard.player_mercy = 0.90;
        agent.blackboard.current_behavior = "Merchant".to_string();
        agent.blackboard.current_mercy_valence = 0.95;
        agent
    }

    /// Guardian NPC — higher health, protective valence.
    pub fn create_guardian(position: Position, patrol_points: Option<Vec<Position>>) -> NpcAgent {
        let mut agent = Self::create_basic(position, patrol_points);
        agent.blackboard.current_health = 150.0;
        agent.blackboard.max_health = 150.0;
        agent.blackboard.current_behavior = "Guardian".to_string();
        agent.blackboard.current_mercy_valence = 0.65;
        agent
    }

    /// Batch spawn for world initialization (simple linear layout).
    pub fn spawn_basic_batch(count: usize, base_pos: Position, spacing: f32) -> Vec<NpcAgent> {
        (0..count)
            .map(|i| {
                let offset = Vector2::new(i as f32 * spacing, 0.0);
                Self::create_basic(base_pos + offset, None)
            })
            .collect()
    }
}