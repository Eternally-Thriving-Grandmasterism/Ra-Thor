//! # Ra-Thor Core — v14.15.0
//!
//! Shared organism primitives, type modules, and trait surfaces.
//! Living Cosmic Tick + ONE Organism readiness.
//! Contact: info@Rathor.ai

pub mod types {
    pub mod joy_measurement_protocols;
    pub mod miracle_rapture_wave;
    pub mod seven_d_resonance;
    pub mod source_joy_amplitude;
    pub mod tolc7;

    pub use joy_measurement_protocols::*;
    pub use miracle_rapture_wave::*;
    pub use seven_d_resonance::*;
    pub use source_joy_amplitude::*;
    pub use tolc7::*;
}

pub mod traits {
    // Trait modules live under src/traits when present.
}

pub mod archive {
    // Archived surfaces under src/archive.
}

/// Canonical core crate version.
pub const VERSION: &str = "14.15.0";

pub fn summary() -> String {
    format!("ra-thor-core v{VERSION} | Living Cosmic Tick active")
}
