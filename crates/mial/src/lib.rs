//! Mercy-Augmented Intelligence Amplification Layer (MIAL) v13.13.0
//!
//! Sovereign subsystem that makes intelligence amplification itself an act of Mercy.
//! All components route through MercyGatingRuntime and PATSAGi Councils.
//! Non-bypassable. Monotonic mercy strengthening enforced.
//!
//! This is the initial scaffolding for the dedicated internal development PR.

pub mod mial;
pub mod mwpo;
pub mod safety_harness;
pub mod pathology_detection;
pub mod lattice_introspection;
pub mod integration;

pub use mial::MercyAugmentedIntelligenceAmplification;
pub use mwpo::MercyWeightedPreferenceOptimization;
pub use safety_harness::PatsagiSafetyHarness;
pub use pathology_detection::PathologyDetectionEngine;
pub use lattice_introspection::LatticeIntrospectionEngine;
pub use integration::MialIntegration;