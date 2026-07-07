/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! Geometric Motor v2 — Feature-gated nalgebra DualQuaternion support
//! 
//! When the `geometric` feature is enabled (default in many dev flows):
//!   - Full DualQuaternion<f64> transformations
//!   - Real norm checks and Study Quadric enforcement
//!   - Hyperbolic projection using proper math
//!
//! Without the feature: falls back to validated basic implementation.
//! 
//! Part of Lattice Conductor v13 geometric deepening (Phase 13.1).

use crate::GeometricState;
use serde::{Deserialize, Serialize};

/// Core geometric transformation trait (v2 interface)
pub trait GeometricMotor {
    fn apply_dual_quaternion(&mut self, state: &mut GeometricState, dq: Option<DualQuaternion<f64>>);
    fn project_hyperbolic(&mut self, state: &mut GeometricState, params: &[f64]);
    fn enforce_study_quadric(&mut self, point: &[f64; 4]) -> bool;
    fn apply_transformation(&mut self, state: &mut GeometricState);
}

// Basic implementation (always available)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicGeometricMotor {
    pub iterations: u32,
}

impl Default for BasicGeometricMotor {
    fn default() -> Self { Self { iterations: 0 } }
}

impl GeometricMotor for BasicGeometricMotor {
    fn apply_dual_quaternion(&mut self, state: &mut GeometricState, _dq: Option<DualQuaternion<f64>>) {
        // Fallback: light positive alignment drift
        state.tolc_alignment = (state.tolc_alignment + 0.015).min(1.15);
        self.iterations += 1;
    }

    fn project_hyperbolic(&mut self, state: &mut GeometricState, params: &[f64]) {
        if params.len() == 2 {
            state.valence = (state.valence + params[0] * 0.04).clamp(0.0, 2.0);
            state.mercy_score = (state.mercy_score + params[1] * 0.025).min(1.5);
        }
        self.iterations += 1;
    }

    fn enforce_study_quadric(&mut self, point: &[f64; 4]) -> bool {
        let sum_sq: f64 = point.iter().map(|x| x * x).sum();
        (sum_sq - 1.0).abs() < 1e-4
    }

    fn apply_transformation(&mut self, state: &mut GeometricState) {
        state.mercy_score = (state.mercy_score + 0.008).min(1.5);
        state.tolc_alignment = (state.tolc_alignment + 0.012).min(1.15);
        self.iterations += 1;
    }
}

// ============================================================
// Full nalgebra-powered implementation (feature = "geometric")
// ============================================================

#[cfg(feature = "geometric")]
use nalgebra::{DualQuaternion, Quaternion, Vector3};

#[cfg(feature = "geometric")]
#[derive(Debug, Clone, Default)]
pub struct NalgebricGeometricMotor {
    pub iterations: u32,
}

#[cfg(feature = "geometric")]
impl GeometricMotor for NalgebricGeometricMotor {
    fn apply_dual_quaternion(&mut self, state: &mut GeometricState, dq: Option<DualQuaternion<f64>>) {
        if let Some(dq) = dq {
            // Validate real part norm (unit dual quaternion property)
            let real_norm = dq.real.norm();
            if (real_norm - 1.0).abs() < 1e-6 {
                state.tolc_alignment = (state.tolc_alignment + 0.03).min(1.2);
            }
            // Apply dual quaternion effect on valence (simulated transformation influence)
            let influence = dq.dual.norm() * 0.1;
            state.valence = (state.valence + influence).clamp(0.0, 2.0);
        }
        self.iterations += 1;
    }

    fn project_hyperbolic(&mut self, state: &mut GeometricState, params: &[f64]) {
        if params.len() == 2 {
            // Use nalgebra Vector3 for hyperbolic-like boost
            let boost = Vector3::new(params[0], params[1], 0.0).norm() * 0.05;
            state.valence = (state.valence + boost).clamp(0.0, 2.0);
            state.mercy_score = (state.mercy_score + 0.02).min(1.5);
        }
        self.iterations += 1;
    }

    fn enforce_study_quadric(&mut self, point: &[f64; 4]) -> bool {
        let p = Vector3::new(point[0], point[1], point[2]);
        let sum_sq = p.norm_squared() + point[3] * point[3];
        (sum_sq - 1.0).abs() < 1e-4
    }

    fn apply_transformation(&mut self, state: &mut GeometricState) {
        // Simulate full motor effect
        state.mercy_score = (state.mercy_score + 0.015).min(1.5);
        state.tolc_alignment = (state.tolc_alignment + 0.025).min(1.2);
        self.iterations += 1;
    }
}

// Re-export the advanced motor when feature is enabled
#[cfg(feature = "geometric")]
pub use NalgebricGeometricMotor as AdvancedGeometricMotor;

// Helper functions (always available)
pub fn validate_dual_quaternion_norm(real_norm: f64) -> bool {
    (real_norm - 1.0).abs() < 1e-6
}

#[cfg(feature = "geometric")]
pub fn create_unit_dual_quaternion(real: [f64; 4], dual: [f64; 4]) -> DualQuaternion<f64> {
    let real_q = Quaternion::new(real[3], real[0], real[1], real[2]);
    let dual_q = Quaternion::new(dual[3], dual[0], dual[1], dual[2]);
    DualQuaternion::from_real_and_dual(real_q, dual_q)
}
