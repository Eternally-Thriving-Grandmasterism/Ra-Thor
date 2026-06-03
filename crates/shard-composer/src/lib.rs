//! shard-composer
//!
//! High-level unification crate for Ra-Thor Sovereign Shards.

pub use ra_thor_mercy as mercy;

#[cfg(feature = "geometric-intelligence")]
pub use geometric_intelligence as geometric;

#[cfg(feature = "real-estate-lattice")]
pub use real_estate_lattice as real_estate;

#[cfg(feature = "quantum-swarm-orchestrator")]
pub use ra_thor_quantum_swarm_orchestrator as quantum_swarm;

#[cfg(feature = "patsagi-councils")]
pub use patsagi_councils as patsagi;

// ONE Organism participation
#[cfg(feature = "one-organism")]
pub mod adapter;

#[cfg(feature = "one-organism")]
pub use adapter::ShardComposerAdapter;
