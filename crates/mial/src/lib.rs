//! Mercy-Augmented Intelligence Amplification Layer (MIAL).
//! The compiled path is `mwpo` and `pathology_detection`.
//! Geometry scores are formulas of source literals.
//! `0.999999` in `evaluate_geometry_mercy_component` is a source literal, not a measured value.
//! These files are not modules: `mial.rs`, `safety_harness.rs`, `lattice_introspection.rs`,
//! `integration.rs`, `mod.rs`.
//! `examples/` and `tests/` are not cargo targets.
//! Contact: info@Rathor.ai

pub mod mwpo;
pub mod pathology_detection;

pub use mwpo::MercyWeightedPreferenceOptimization;
pub use pathology_detection::PathologyDetectionEngine;
