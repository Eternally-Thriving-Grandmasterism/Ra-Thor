//! crates/powrush/src/lib.rs
//! Powrush — Post-Scarcity RBE MMO + v15 Hybrid NPC AI + Clifford Healing Fields
//! ONE Organism aligned | TOLC 8 Mercy Gates | AG-SML v1.0
//! v16.0 update: PowrushMMOWorld authoritative simulation + chunk persistence

pub mod clifford_healing_fields;
pub mod npc;           // v15 Hybrid NPC AI System
pub mod simulation;    // High-level world simulation / game loop wiring (now v16 PowrushMMO)
pub mod economy;       // RBE Economy + Crafting System (v15.6+)

// Re-exports for convenience
pub use clifford_healing_fields::{
    CliffordHealingField,
    HealingConfig,
    HealingFieldError,
    GlobalCoherence,
    OrganismField,
    demo_multi_organism_healing,
};

pub use npc::{
    NpcAgent,
    NpcBlackboard,
    NpcFactory,
    NpcIntegration,
    NpcSystem,
    PatrolManager,
    PatrolPath,
    PatrolState,
    PerceptionSystem,
    SpatialHash,
    UtilityAction,
};

pub use simulation::{
    WorldSimulation,
    PowrushMMOWorld,      // Authoritative MMO world entrypoint (type alias)
    WorldChunk,           // Chunk persistence + resource regen
    SessionSyncStub,      // Networking/session sync stub
};

pub use economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
