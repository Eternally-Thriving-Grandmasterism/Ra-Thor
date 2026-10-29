//! crates/powrush/src/npc/mod.rs
//! v15 Hybrid NPC AI System
//! Blackboard + Consideration + Utility + Spatial + Relationship + Dialogue + Epigenetic
//! Mercy as first-class architectural principle | ONE Organism aligned | AG-SML v1.0

// === Core Data Layer ===
pub mod blackboard;

// === Decision & Utility Layer ===
pub mod consideration;
pub mod behavior;

// === Perception & Spatial Layer ===
pub mod perception;
pub mod spatial_hash;
pub mod patrol;

// === Social & Relationship Layer ===
pub mod relationship;
pub mod dialogue;

// === Epigenetic / RBE Layer ===
pub mod epigenetic;

// === High-Level Systems ===
pub mod system;
pub mod npc_integration;
pub mod npc_spawning;

// === Clean Re-exports ===
pub use blackboard::{NpcBlackboard, Position, BlackboardKey, BlackboardValue};

pub use consideration::{
    Consideration,
    MercyAlignmentConsideration,
    HealthConsideration,
    PlayerThreatConsideration,
    PostScarcityConsideration,
    PlayerAscensionConsideration,
};

pub use behavior::{NpcAgent, UtilityAction};

pub use perception::PerceptionSystem;
pub use spatial_hash::SpatialHash;
pub use patrol::{PatrolManager, PatrolState, PatrolPath};

pub use relationship::{Relationship, RelationshipLevel};
pub use dialogue::{DialogueSystem, DialogueResponse, DialogueTone};

pub use system::NpcSystem;
pub use npc_integration::NpcIntegration;
pub use npc_spawning::NpcFactory;

pub use epigenetic::distribute_epigenetic_blessing;
