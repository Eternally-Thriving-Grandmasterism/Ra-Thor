//! crates/powrush/src/lib.rs
//! Powrush — Post-Scarcity RBE MMO + v15 Hybrid NPC AI + Clifford Healing Fields
//! ONE Organism aligned | TOLC 8 Mercy Gates | AG-SML v1.0

pub mod clifford_healing_fields;
pub mod npc;           // v15 Hybrid NPC AI System
pub mod simulation;    // High-level world simulation / game loop wiring
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

pub use simulation::WorldSimulation;

pub use economy::{RbeEconomy, CraftingRecipe, get_default_recipes};