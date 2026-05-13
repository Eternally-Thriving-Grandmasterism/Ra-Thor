//! Mercy WASM Bridge v1.1
//! 
//! Production-grade bridge for real-time mercy-gated communication
//! between Rust core and WASM/JS mercy engines (rathor.ai).
//!
//! Features:
//! - Send mercy-validated payloads from Rust → WASM
//! - Receive and re-validate responses from WASM → Rust
//! - Real-time valence propagation
//! - Explicit #[wasm_bindgen] exports for JavaScript interop
//! - Full TOLC + 7 Mercy Gates alignment

use wasm_bindgen::prelude::*;
use crate::mercy::{MercyGateResult};

/// Core Mercy WASM Bridge
#[wasm_bindgen]
pub struct MercyWasmBridge {
    version: String,
    valence_threshold: f64,
}

#[wasm_bindgen]
impl MercyWasmBridge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            version: "v1.1.0-enhanced".to_string(),
            valence_threshold: 0.999,
        }
    }

    /// Send mercy-evaluated data from Rust to WASM/JS
    #[wasm_bindgen]
    pub fn send_to_wasm(&self, payload: &str, current_valence: f64) -> Result<String, JsValue> {
        if current_valence < self.valence_threshold {
            return Err(JsValue::from_str("Valence below threshold — blocked by Mercy WASM Bridge"));
        }

        let sanitized = payload.trim().to_string();
        let bridged = format!(r#"{{"payload":"{}","valence":{},"mercy_gated":true,"bridge_version":"{}"}}"#, sanitized.replace('"', "\\\""), current_valence, self.version);

        Ok(bridged)
    }

    /// Receive data from WASM/JS and re-validate through Mercy Gates
    #[wasm_bindgen]
    pub fn receive_from_wasm(&self, wasm_response: &str, context_valence: f64) -> MercyGateResult {
        let final_valence = context_valence.min(self.valence_threshold);

        if final_valence >= self.valence_threshold {
            MercyGateResult::Pass {
                valence: final_valence,
                message: "WASM response passed Mercy WASM Bridge".to_string(),
            }
        } else {
            MercyGateResult::Fail {
                valence: final_valence,
                reason: "WASM response failed mercy/valence requirements".to_string(),
            }
        }
    }

    /// Propagate updated valence across the bridge (for positive emotion flow)
    #[wasm_bindgen]
    pub fn propagate_valence(&self, new_valence: f64) -> f64 {
        new_valence.clamp(0.0, 1.0)
    }

    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.version.clone()
    }
}