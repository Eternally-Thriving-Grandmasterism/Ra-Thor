//! powrush — policy hint emission for the Ra-Thor side of the Powrush bridge.
//!
//! The default build is `governance_types` and `policy_hint_emission`
//! (schema `ra_thor_policy_hint_v1`). A hint that fails validation is not written.
//! This crate does not run an MMO, and it does not apply hints to a live world.
//!
//! Sources that need bevy or nalgebra, and the stub game modules, stay on disk.
//! They are not `mod`ded here. `src/bin` is not a cargo target.
//! Contact: info@Rathor.ai

pub mod governance_types;
pub mod policy_hint_emission;

pub use governance_types::{
    AscensionLevel, Faction, MercyGateStatus, Player, PlayerNeeds, PowrushGame, ResourceType,
    WorldResource,
};

// Dual-repo soft feedback emission (Ra-Thor → Powrush-MMO)
pub use policy_hint_emission::{
    conservative_mercy_presence, emit_after_deliberation, emit_policy_hints, EmissionError,
    PolicyHint, PolicyHintEnvelope, CLOSED_CATEGORIES, DEFAULT_EMISSION_PATH, SCHEMA_V1,
};
