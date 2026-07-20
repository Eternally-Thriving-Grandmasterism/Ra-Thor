//! crates/powrush/src/lib.rs — v14.15.0
//! Powrush RBE + governance compatibility surface for PATSAGi Councils.
//! Living Cosmic Tick + ONE Organism readiness. Contact: info@Rathor.ai

pub mod clifford_healing_fields;
pub mod npc;
pub mod simulation;
pub mod economy;
pub mod powrush_mmo_core;
pub mod device_capability;
pub mod experience_tier;
pub mod ui_adaptation;
pub mod virtual_joystick;
pub mod governance_types;

pub use device_capability::{DeviceCapability, Platform, InputMethod, DeviceCapabilityPlugin};
pub use experience_tier::{ExperienceTier, ExperienceTierPlugin};
pub use ui_adaptation::{UiAdaptation, UiAdaptationPlugin};
pub use virtual_joystick::{
    VirtualJoystick, VirtualJoystickPlugin, JoystickMoveEvent, PredictedState,
    ReplayLogger, DeterministicReplayLog, ReplayEntry,
    MerkleTree, MerkleProof,
    TargetedSparseMerkleTree, TargetedSparseProof, compute_state_checksum,
};
pub use simulation::{Position, predict_move_position};

// PATSAGi / WorldGovernance closed-loop surface
pub use governance_types::{
    AscensionLevel, Faction, MercyGateStatus, Player, PlayerNeeds, PowrushGame, ResourceType,
    WorldResource,
};
