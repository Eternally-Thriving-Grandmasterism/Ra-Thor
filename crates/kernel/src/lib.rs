//! `kernel` is a standalone workspace. It is not a root member.
//!
//! Compiled surface: `PACKAGE_VERSION_SOURCE` and `STANDALONE_BOUNDARY`.
//! `PACKAGE_VERSION_SOURCE` is the package version string `14.15.0`.
//! That string is not a readiness measurement.
//! `STANDALONE_BOUNDARY` records the dependency boundary of this crate.
//! That string is not a measured result.
//!
//! `lattice_v14_boot` is not a module. `lattice-conductor-v14` is a root member.
//! It does not build as its own workspace. This card does not edit v14.
//! `enforce_cosmic_loop_on_boot` and `arbitration_rejects_disable` stay in that file.
//!
//! These files are not modules: `../lib.rs`, `absolute_truth_enforcer.rs`,
//! `fenca.rs`, `meta_orchestrator.rs`, `permanence_code_loop.rs`,
//! `root_core_orchestrator.rs`, `sub_core_trait.rs`, `tolc_core_enforcer.rs`,
//! `wasm_boot.rs`.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

/// Package version string from this crate's manifest. Not a readiness measurement.
pub const PACKAGE_VERSION_SOURCE: &str = "14.15.0";

/// Source string for the dependency boundary of this standalone crate.
pub const STANDALONE_BOUNDARY: &str =
    "lattice-conductor-v14 is not a path dependency; v14 boot is not compiled";
