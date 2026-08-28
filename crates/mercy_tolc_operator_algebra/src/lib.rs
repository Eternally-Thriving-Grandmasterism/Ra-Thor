//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//!
//! ## Ambient · valence · adaptive floor · concurrent zones · soft feedback · LatticeHealthReport · adaptive Cosmic Tick · NEVC · Tikhonov-damped projector
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

mod soft_feedback;
pub use soft_feedback::*;

mod nevc;
pub use nevc::*;

include!("algebra.rs");

#[cfg(test)]
mod tests {
    include!("algebra_tests.rs");
    include!("algebra_tests_restored.rs");
}
