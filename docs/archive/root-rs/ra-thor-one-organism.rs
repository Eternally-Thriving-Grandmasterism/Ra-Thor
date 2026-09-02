//! DEPRECATED ROOT ENTRY — do not extend.
//!
//! **Source of truth:** `crates/ra-thor-one-organism` (v14.15+)
//!
//! ```toml
//! ra-thor-one-organism = { path = "crates/ra-thor-one-organism" }
//! # optional live paths:
//! # features = ["extended-live", "web-demo"]
//! ```
//!
//! ```rust
//! use ra_thor_one_organism::{launch_one_organism_core, OneOrganismCore};
//! let mut core = launch_one_organism_core();
//! assert!(core.is_cosmic_loop_ready());
//! ```
//!
//! Historical root implementations (compatibility guardians, dual surface)
//! were retired to prevent drift from the real path-dependent crate.
//! See: docs/ONE_ORGANISM_LATTICE_CONDUCTOR_V14_WIRING.md, TIER_MAP.md, PRODUCTION_READINESS.md
//!
//! Contact: info@Rathor.ai | Cosmic Loop is MANDATORY IDENTITY.

compile_error!(
    "ra-thor-one-organism.rs at repo root is retired. \
     Use crates/ra-thor-one-organism (path dependency). \
     See TIER_MAP.md and PRODUCTION_READINESS.md."
);
