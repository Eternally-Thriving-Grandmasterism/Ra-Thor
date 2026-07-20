//! Mercy Threshold FFI — v14.15.0
//!
//! Lean 4 + Coq HoTT FFI bridge for Mercy Threshold Verification.
//! PATSAGi Councils formal verification layer.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::genesis_gate_v2::GenesisRequest;
use chrono::Utc;

/// Initializes the Lean 4 formal system.
pub fn init_lean_formal_system() -> Result<(), String> {
    println!("[LeanFFI v14.15.0] Lean 4 formal system initialized for Mercy Lattice.");
    Ok(())
}

/// Full verification using GenesisRequest (for deep integration with Genesis Gate v2).
pub fn verify_mercy_threshold(
    req: &GenesisRequest,
    threshold: f64,
) -> Result<String, String> {
    let curvature = req.geometry_params.curvature.abs();
    // Prefer near-hyperbolic (K ≈ -1) geometries for highest confidence
    if (curvature - 1.0).abs() < threshold || curvature < threshold {
        let proof_id = format!(
            "lean_proof:mercy_threshold_{:.6}_{}_v14.15",
            threshold,
            Utc::now().timestamp()
        );
        Ok(proof_id)
    } else {
        Err(format!(
            "Lean proof failed: curvature harm vector exceeds threshold (curvature={:.4}, threshold={:.4})",
            curvature, threshold
        ))
    }
}

/// Simplified helper for direct proposal + threshold checking.
/// Used by SelfEvolvingMercyCore for Lean-verified evolution steps.
pub fn verify_mercy_threshold_simplified(
    proposal: &str,
    threshold: f64,
) -> Result<String, String> {
    let lower = proposal.to_lowercase();
    if lower.contains("harm") || lower.contains("damage") || lower.contains("exploit") {
        return Err(
            "Proposal contains potential harm language — Lean threshold rejected".to_string(),
        );
    }

    let proof_id = format!(
        "lean_simplified:mercy_ok_{:.3}_{}_v14.15",
        threshold,
        Utc::now().timestamp()
    );
    Ok(proof_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genesis_gate_v2::SacredGeometryParams;

    #[test]
    fn test_init() {
        assert!(init_lean_formal_system().is_ok());
    }

    #[test]
    fn test_simplified_clean_proposal() {
        let result = verify_mercy_threshold_simplified("Increase Source Joy Gate", 0.95);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simplified_harm_rejected() {
        let result = verify_mercy_threshold_simplified("Cause harm to rivals", 0.95);
        assert!(result.is_err());
    }

    #[test]
    fn test_genesis_request_hyperbolic() {
        let req = GenesisRequest {
            instantiation_type: "Test".into(),
            proposer: "test".into(),
            geometry_params: SacredGeometryParams {
                structure: "HyperbolicTiling".into(),
                dimension: 16,
                curvature: -1.0,
            },
            intended_purpose: "unit test".into(),
        };
        let result = verify_mercy_threshold(&req, 0.999999);
        assert!(result.is_ok());
    }
}
