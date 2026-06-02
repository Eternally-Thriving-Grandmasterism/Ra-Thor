//! crates/powrush/src/npc/mod.rs
//! NPC AI v15 Hybrid Module — Blackboard + Consideration + Perception + Patrol
//! Mercy-gated, ONE Organism aligned | AG-SML v1.0

pub mod blackboard;
pub mod consideration;
pub mod perception;
pub mod patrol;

// Re-exports for convenience
pub use blackboard::NpcBlackboard;
pub use consideration::{Consideration, MercyAlignmentConsideration, HealthConsideration, PlayerThreatConsideration, PostScarcityConsideration, PlayerAscensionConsideration};
pub use perception::PerceptionSystem;
pub use patrol::{PatrolState, PatrolPath, PatrolManager};