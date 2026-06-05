//! Powrush Crate — Living Simulation Core (v14.6.1)
//!
//! Professional wiring of the approved next natural steps into the Ra-Thor monorepo.
//! The living simulation where humans, AI, and AGI coexist, learn, and earn together
//! in joy and abundance is now running in code.
//!
//! CUSTOM PROPRIETARY MMORPG SYSTEMS (Ra-Thor Native, zero external licensing):

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "client")]
pub mod client;

pub mod common;

// Re-exports for Ra-Thor lattice integration
pub use ra_thor_one_organism::RaThorOneOrganism;
pub use self_evolution_gate::SelfEvolutionGate;

// Existing content preserved and extended with server/client separation
