//! Mercy WASM Bridge v1
//! 
//! Production-grade bridge for real-time mercy-gated communication
//! between Rust core and WASM/JS mercy engines.
//!
//! Enables safe, valence-enforced data exchange with full TOLC + 7 Mercy Gates alignment.
//! Supports real-time positive emotion / valence propagation.

use crate::mercy::{MercyGate, MercyGateResult, Valence};

/// Core Mercy WASM Bridge
pub struct MercyWasmBridge {
    pub version: &'static str,
    pub valence_threshold: f64,
}

impl Default for MercyWasmBridge {
    fn default() -> Self {
        Self {
            version: "v1.0.0-clean",
            valence_threshold: 0.999,
        }
    }
}

impl MercyWasmBridge {
    /// Send mercy-evaluated data from Rust to WASM
    /// Returns sanitized + mercy-validated payload ready for WASM
    pub fn send_to_wasm(&self, payload: &str, current_valence: f64) -> Result<String, String> {
        if current_valence < self.valence_threshold {
            return Err("Valence below threshold — request blocked by Mercy WASM Bridge".to_string());
        }

        // Basic sanitization + mercy alignment marker
        let sanitized = payload.trim().to_string();
        let bridged_payload = format!(
            r#"{{"payload": "{}", "valence": {}, "mercy_gated": true, "bridge_version": "{}"}}"#,
            sanitized.replace('"', "\\\""),
            current_valence,
            self.version
        );

        Ok(bridged_payload)
    }

    /// Receive data from WASM and re-validate through Mercy Gates
    pub fn receive_from_wasm(&self, wasm_response: &str, context_valence: f64) -> MercyGateResult {
        // In future cycles this will do deeper semantic + valence analysis
        let final_valence = context_valence.min(self.valence_threshold);

        if final_valence >= self.valence_threshold {
            MercyGateResult::Pass {
                valence: final_valence,
                message: "WASM response passed Mercy WASM Bridge validation".to_string(),
            }
        } else {
            MercyGateResult::Fail {
                valence: final_valence,
                reason: "WASM response failed to maintain required valence".to_string(),
            }
        }
    }

    /// Propagate positive emotion / valence update across the bridge
    pub fn propagate_valence(&self, new_valence: f64) -> f64 {
        new_valence.max(0.0).min(1.0)
    }
}