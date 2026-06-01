//! shard-composer
//!
//! High-level unification crate for Ra-Thor Sovereign Shards.
//!
//! This crate provides convenient feature flags to compose different
//! shard profiles from the monorepo.
//!
//! ## Available Profiles
//!
//! - `full` — Complete ONE Organism
//! - `focused-real-estate` / `real-estate` — Real Estate focused (includes Professional Judgment Layer)
//! - `focused-geometry` / `geometry` — Geometry + Riemannian focused
//!
//! ## Usage with xtask (Recommended)
//!
//! ```bash
//! cargo xtask build-shard --profile focused-real-estate
//! cargo xtask build-shard --profile full --release
//! ```
//!
//! ## Direct Usage
//!
//! ```bash
//! cargo build -p shard-composer --features focused-real-estate
//! ```

pub use ra_thor_mercy as mercy;

#[cfg(feature = "geometric-intelligence")]
pub use geometric_intelligence as geometric;

#[cfg(feature = "real-estate-lattice")]
pub use real_estate_lattice as real_estate;

#[cfg(feature = "quantum-swarm-orchestrator")]
pub use ra_thor_quantum_swarm_orchestrator as quantum_swarm;

#[cfg(feature = "patsagi-councils")]
pub use patsagi_councils as patsagi;
