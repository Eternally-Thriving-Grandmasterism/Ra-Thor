//! Mercy Threshold Checker — v14.15.0
//!
//! Production mercy threshold with optional Lean 4 FFI (lean-sys).
//! When `verified-mercy` is enabled, calls machine-checked Lean proofs.
//! Otherwise falls back to pure-Rust logic matching the Lean specification.
//!
//! Living Cosmic Tick aligned. Used by Genesis Gate v2 and Self-Evolving Mercy Core.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use tracing::{info, warn};

#[cfg(feature = "verified-mercy")]
use lean_sys::lean_init;

/// FFI declarations matching MercyThresholdExport.lean
#[cfg(feature = "verified-mercy")]
extern "C" {
    fn rathor_geometry_init() -> u32;
    fn mercy_threshold_safe(score_high: f64, mercy_valence: f64) -> bool;
    fn verified_mercy_check_ffi(
        score_high: f64,
        mercy_valence: f64,
        request_id: u64,
    ) -> *mut lean_sys::lean_object;
    fn rathor_geometry_shutdown();
}

/// Production Mercy Threshold Checker.
pub struct MercyThresholdChecker {
    initialized: bool,
}

impl MercyThresholdChecker {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Initialize Lean 4 runtime (must be called before any verified checks).
    pub fn initialize(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(feature = "verified-mercy")]
        {
            unsafe {
                lean_init();
                let status = rathor_geometry_init();
                if status != 0 {
                    return Err("Lean 4 geometry FFI initialization failed".to_string());
                }
            }
            info!("[MercyThreshold v14.15.0] Lean 4 verified mercy runtime initialized");
        }

        #[cfg(not(feature = "verified-mercy"))]
        {
            info!("[MercyThreshold v14.15.0] Fallback pure-Rust threshold active (no Lean FFI)");
        }

        self.initialized = true;
        Ok(())
    }

    /// Core verified check — calls Lean 4 proof when feature enabled.
    pub fn check(
        &self,
        score_high: f64,
        mercy_valence: f64,
        request_id: u64,
    ) -> Result<bool, String> {
        if !self.initialized {
            return Err("MercyThresholdChecker not initialized".to_string());
        }

        #[cfg(feature = "verified-mercy")]
        {
            unsafe {
                let result = mercy_threshold_safe(score_high, mercy_valence);
                if result {
                    info!(
                        "[SUCCESS] Request {} passed verified mercy threshold (score_high={:.4})",
                        request_id, score_high
                    );
                } else {
                    warn!(
                        "[REJECT] Request {} failed verified mercy threshold (score_high={:.4})",
                        request_id, score_high
                    );
                }
                Ok(result)
            }
        }

        #[cfg(not(feature = "verified-mercy"))]
        {
            // Fallback pure-Rust logic (matches Lean specification)
            let result = score_high > 0.95 && (mercy_valence - 1.0).abs() < f64::EPSILON;
            if result {
                info!("[FALLBACK SUCCESS] Request {}", request_id);
            } else {
                warn!("[FALLBACK REJECT] Request {}", request_id);
            }
            Ok(result)
        }
    }

    /// Full end-to-end verified check with structured message.
    pub fn verified_check_full(
        &self,
        score_high: f64,
        mercy_valence: f64,
        request_id: u64,
    ) -> Result<(bool, String), String> {
        if !self.initialized {
            return Err("MercyThresholdChecker not initialized".to_string());
        }

        #[cfg(feature = "verified-mercy")]
        {
            unsafe {
                let _io_result = verified_mercy_check_ffi(score_high, mercy_valence, request_id);
                let passed = mercy_threshold_safe(score_high, mercy_valence);
                let msg = if passed {
                    format!(
                        "[SUCCESS] Request {} — TOLC 8 safe via Lean 4 proof | Living Cosmic Tick",
                        request_id
                    )
                } else {
                    format!("[REJECT] Request {} failed mercy threshold", request_id)
                };
                Ok((passed, msg))
            }
        }

        #[cfg(not(feature = "verified-mercy"))]
        {
            let passed = score_high > 0.95 && (mercy_valence - 1.0).abs() < f64::EPSILON;
            let msg = if passed {
                format!(
                    "[FALLBACK PASS] Request {} | Living Cosmic Tick",
                    request_id
                )
            } else {
                format!("[FALLBACK REJECT] Request {}", request_id)
            };
            Ok((passed, msg))
        }
    }

    /// Lightweight readiness summary.
    pub fn summary(&self) -> String {
        format!(
            "MercyThresholdChecker v14.15.0 | initialized={} | mode={}",
            self.initialized,
            if cfg!(feature = "verified-mercy") {
                "Lean-verified"
            } else {
                "pure-Rust fallback"
            }
        )
    }
}

impl Default for MercyThresholdChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MercyThresholdChecker {
    fn drop(&mut self) {
        #[cfg(feature = "verified-mercy")]
        unsafe {
            rathor_geometry_shutdown();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_mercy_threshold_j27_sovereignty() {
        let mut checker = MercyThresholdChecker::new();
        checker.initialize().expect("init failed");

        let (passed, msg) = checker.verified_check_full(0.992, 1.0, 27).unwrap();
        assert!(passed, "J27 should pass mercy threshold");
        assert!(msg.contains("SUCCESS") || msg.contains("PASS"));
    }

    #[test]
    fn test_verified_mercy_threshold_j84_infinite() {
        let mut checker = MercyThresholdChecker::new();
        checker.initialize().expect("init failed");

        let (passed, _) = checker.verified_check_full(0.987, 1.0, 84).unwrap();
        assert!(passed);
    }

    #[test]
    fn test_rejection_below_threshold() {
        let mut checker = MercyThresholdChecker::new();
        checker.initialize().expect("init failed");

        let (passed, _) = checker.verified_check_full(0.91, 1.0, 999).unwrap();
        assert!(!passed);
    }

    #[test]
    fn test_fallback_when_feature_disabled() {
        let mut checker = MercyThresholdChecker::new();
        let _ = checker.initialize();
        let (passed, _) = checker.verified_check_full(0.96, 1.0, 1).unwrap();
        assert!(passed);
    }

    #[test]
    fn test_summary() {
        let checker = MercyThresholdChecker::new();
        assert!(checker.summary().contains("14.15.0"));
    }
}
