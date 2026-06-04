//! crates/powrush/src/lib.rs
//! Powrush — Post-Scarcity RBE MMO + v15 Hybrid NPC AI + Clifford Healing Fields

pub mod clifford_healing_fields;
pub mod npc;
pub mod simulation;
pub mod economy;
pub mod powrush_mmo_core;
pub mod device_capability;
pub mod experience_tier;
pub mod ui_adaptation;

pub use device_capability::{DeviceCapability, Platform, InputMethod, DeviceCapabilityPlugin};
pub use experience_tier::{ExperienceTier, ExperienceTierPlugin};
pub use ui_adaptation::{UiAdaptation, UiAdaptationPlugin};
