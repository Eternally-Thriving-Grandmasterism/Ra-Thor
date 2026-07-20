//! v13 compatibility surface for lattice-conductor-v14
//!
//! Enabled with `features = ["v13-compat"]`.
//! Additive only — does not alter Cosmic Loop identity or adaptive modulation.
//!
//! See `MIGRATION_v13_to_v14.md`.
//! Contact: info@Rathor.ai

pub mod traits;
pub mod geometric;
pub mod simple;

pub use traits::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use geometric::{GeometricState, MercyWeightedVote};
pub use simple::SimpleLatticeConductor;
