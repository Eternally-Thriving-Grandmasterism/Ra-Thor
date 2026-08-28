//! # mercy_tolc_operator_algebra
//!
//! Executable Living Mercy operator algebra for the Ra-Thor lattice under TOLC 8.
//! Tikhonov-damped projector P_λ = E(EᵀE + λI)⁻¹Eᵀ landed 0.5.19.
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

#![forbid(unsafe_code)]

include!("algebra_impl.rs");
include!("algebra_tests.rs");
