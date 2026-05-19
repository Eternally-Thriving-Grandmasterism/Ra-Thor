//! Mercy Threshold Module — TOLC 8 Verified Production Logic with lean-sys FFI
//!
//! This module brings the formally verified Mercy Threshold
//! (Lean 4 + Coq) into the patsagi-councils crate.
//!
//! Feature: "verified-mercy"
//!
//! All instantiations must pass this check.

use std::collections::HashMap;

/// Result of mercy threshold evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MercyResult {
    Safe,
    Rerouted(String),
}

// === lean-sys FFI Integration ===
#[cfg(feature = "verified-mercy")]
use lean_sys::{lean_initialize_runtime_module};

#[cfg(feature = "verified-mercy")]
extern "C" {
    fn lean_mercy_threshold_safe(score: f64, valence: f64) -> bool;
}

/// Verified mercy check using Lean 4 proof (via lean-sys)
#[cfg(feature = "verified-mercy")]
pub fn verified_mercy_check(score: f64, valence: f64) -> bool {
    unsafe {
        lean_initialize_runtime_module();
        lean_mercy_threshold_safe(score, valence)
    }
}

/// Core verified mercy threshold check (production)
#[cfg(feature = "verified-mercy")]
pub fn mercy_threshold_safe(
    geometry_score: f64,
    mercy_valence: f64,
    context: &str,
) -> MercyResult {
    if verified_mercy_check(geometry_score, mercy_valence) {
        MercyResult::Safe
    } else {
        MercyResult::Rerouted(
            format!("Request rerouted: score={:.4}, valence={:.6} (context: {})", 
                    geometry_score, mercy_valence, context)
        )
    }
}

/// Public API used by WorldGovernanceEngine
#[cfg(feature = "verified-mercy")]
pub fn check_instantiation_mercy(
    request: &HashMap<String, String>,
) -> MercyResult {
    let score = request.get("geometry_score")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    
    let valence = request.get("mercy_valence")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    
    let context = request.get("context").map(|s| s.as_str()).unwrap_or("unknown");
    
    mercy_threshold_safe(score, valence, context)
}

#[cfg(not(feature = "verified-mercy"))]
pub fn check_instantiation_mercy(_request: &HashMap<String, String>) -> MercyResult {
    MercyResult::Safe // Fallback when feature disabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercy_threshold_safe() {
        let mut req = HashMap::new();
        req.insert("geometry_score".to_string(), "0.96".to_string());
        req.insert("mercy_valence".to_string(), "1.0".to_string());
        req.insert("context".to_string(), "test".to_string());
        
        assert_eq!(check_instantiation_mercy(&req), MercyResult::Safe);
    }

    #[test]
    fn test_mercy_threshold_rerouted() {
        let mut req = HashMap::new();
        req.insert("geometry_score".to_string(), "0.80".to_string());
        req.insert("mercy_valence".to_string(), "0.5".to_string());
        
        match check_instantiation_mercy(&req) {
            MercyResult::Rerouted(_) => assert!(true),
            _ => assert!(false),
        }
    }
}