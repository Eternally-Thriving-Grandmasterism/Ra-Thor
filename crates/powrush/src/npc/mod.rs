//! crates/powrush/src/npc/mod.rs
//! General-purpose NPC / AGiNPC AI infrastructure for Powrush
//! v14.0.0 | MIAL + MWPO + TOLC 8 Mercy Gates enforced
//! AG-SML v1.0

pub mod stats;
pub mod behavior;
pub mod perception;
pub mod patrol;

pub use stats::NpcStats;
pub use behavior::{BehaviorTree, NodeStatus, BehaviorNode};
pub use perception::NpcPerception;
pub use patrol::NpcPatrol;

// Re-export core NPC agent
pub use crate::npc::NpcAgent;

/// Core NPC agent — usable by any intelligent entity
#[derive(Debug)]
pub struct NpcAgent {
    pub stats: NpcStats,
    pub behavior: BehaviorTree,
    pub perception: NpcPerception,
    pub patrol: Option<NpcPatrol>,
    // Future: relationship, dialogue, etc.
}

impl NpcAgent {
    pub fn new(template_id: &str) -> Self {
        Self {
            stats: NpcStats::from_template(template_id),
            behavior: BehaviorTree::default_mercy_aware(),
            perception: NpcPerception::new(),
            patrol: None,
        }
    }

    /// MIAL/MWPO mercy-gated tick
    pub fn tick(&mut self, world_mercy_valence: f64) -> NodeStatus {
        // MWPO mercy-weighted decision
        if world_mercy_valence < 0.999999 {
            self.behavior.apply_mercy_refinement();
        }
        self.behavior.tick(self)
    }
}
