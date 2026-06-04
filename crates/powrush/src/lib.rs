//! crates/powrush/src/lib.rs

pub mod clifford_healing_fields;
pub mod npc;
pub mod simulation;
pub mod economy;
pub mod powrush_mmo_core;
pub mod device_capability;
pub mod experience_tier;
pub mod ui_adaptation;
pub mod virtual_joystick;

pub use device_capability::{DeviceCapability, Platform, InputMethod, DeviceCapabilityPlugin};
pub use experience_tier::{ExperienceTier, ExperienceTierPlugin};
pub use ui_adaptation::{UiAdaptation, UiAdaptationPlugin};
pub use virtual_joystick::{VirtualJoystick, VirtualJoystickPlugin, JoystickMoveEvent};
pub use simulation::{Position, predict_move_position}; // shared for client prediction + reconciliation
