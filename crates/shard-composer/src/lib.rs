//! shard-composer
//!
//! High-level feature unification crate for Ra-Thor Sovereign Shards.
//!
//! This crate provides convenient feature flags that compose multiple
//! underlying crates into well-defined shard profiles:
//!
//! - `full`         → Complete ONE Organism experience
//! - `focused-real-estate` → Lightweight Real Estate focused shard
//! - `focused-geometry`    → Geometry / sacred geometry only
//!
//! Usage:
//! ```bash
//! cargo build -p shard-composer --features focused-real-estate
//! cargo build -p shard-composer --features full
//! ```
//!
//! This is the recommended entry point for building specific Sovereign Shards.

// Re-export key types for convenience when using the unified shard
pub use ra_thor_mercy as mercy;

#[cfg(feature = "geometric-intelligence")]
pub use geometric_intelligence as geometric;

#[cfg(feature = "real-estate-lattice")]
pub use real_estate_lattice as real_estate;

#[cfg(feature = "quantum-swarm-orchestrator")]
pub use ra_thor_quantum_swarm_orchestrator as quantum_swarm;
