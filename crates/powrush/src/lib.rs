//! crates/powrush/src/lib.rs
//! Powrush — Post-Scarcity RBE MMO + v15 Hybrid NPC AI + Clifford Healing Fields
//! ONE Organism aligned | TOLC 8 Mercy Gates | AG-SML v1.0
//! v16.0 update: PowrushMMOWorld authoritative simulation + chunk persistence

pub mod clifford_healing_fields;
pub mod npc;           // v15 Hybrid NPC AI System
pub mod simulation;    // High-level world simulation / game loop wiring (now v16 PowrushMMO)
pub mod economy;       // RBE Economy + Crafting System (v15.6+)
pub mod powrush_mmo_core; // Production-grade MMO Core
pub mod device_capability; // Cross-device capability detection (Option 1 foundation)
pub mod experience_tier;   // Adaptive experience tiers: MobileTown vs DesktopFull

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
    PowrushMMOWorld,
    WorldChunk,
    SessionSyncStub,
};

pub use economy::{RbeEconomy, CraftingRecipe, get_default_recipes};

pub use powrush_mmo_core::{
    PowrushMMOWorld,
    WorldChunk,
    PlayerSession,
    EpigeneticState,
    RBEEconomySimulator,
    EntitySnapshot,
};

pub use device_capability::{DeviceCapability, Platform, InputMethod, DeviceCapabilityPlugin};
pub use experience_tier::{ExperienceTier, ExperienceTierPlugin};
