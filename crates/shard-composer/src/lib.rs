//! shard-composer
//!
//! `ShardComposerAdapter` records blessing arithmetic on
//! `ra-thor-quantum-swarm-orchestrator`.
//! A fresh adapter stores valence `0.0`.
//! `VALENCE_SOURCE_CEILING` is a source ceiling, not a measured valence.
//! Signed files use HMAC-SHA256. The base is `RA_THOR_HMAC_BASE_KEY` and must be at least 32 bytes.
//! This source has no signing key. If that variable is unset, empty, shorter than 32 bytes,
//! or the removed documentation placeholder, `save_to_file` returns an error
//! and `load_from_file` does not accept the signed payload.
//! `save_to_file` does not replace an existing file that fails verification under the current base.
//! `GodlyIntelligenceCoherence` keeps its name. The four fields are recorded impacts, not a measured coherence.
//! `patsagi-councils` is re-exported as `patsagi`. This package is not `crates/council`.
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//!
//! Not path dependencies: `geometric-intelligence` (no library), `real-estate-lattice`
//! (not a standalone build), `crates/mercy` (root member).
//! This package defines no `full` feature. `quantum-swarm-orchestrator` has no `full` feature.
//! `patsagi-councils` has no `full` feature. The re-export uses the public API that builds.
//! Contact: info@Rathor.ai

pub mod adapter;

pub use adapter::{ShardComposerAdapter, VALENCE_SOURCE_CEILING};
pub use patsagi_councils as patsagi;
pub use ra_thor_quantum_swarm_orchestrator as quantum_swarm;
