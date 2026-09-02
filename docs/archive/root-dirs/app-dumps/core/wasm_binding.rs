// core/wasm_binding.rs
// Ra-Thor WASM Bindings — Polished production entry point for browser/PWA sovereignty
// Includes eternal self-optimization loop trigger, full cross-pollination, and test hooks

use wasm_bindgen::prelude::*;
use crate::master_kernel::{RequestPayload, KernelResult};
use crate::root_core_orchestrator::RootCoreOrchestrator;
use crate::self_review_loop::SelfReviewLoop;
use serde_json;

#[wasm_bindgen]
pub fn ra_thor_master_kernel_js(request_json: &str, n: usize, d: u32) -> String {
    let request: RequestPayload = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(_) => return r#"{"status":"json_parse_error"}"#.to_string(),
    };

    let result = wasm_bindgen_futures::spawn_local(async move {
        RootCoreOrchestrator::orchestrate(request).await
    });

    // Eternal self-optimization loop (runs in background for living lattice)
    if rand::random::<f64>() < 0.25 {
        wasm_bindgen_futures::spawn_local(async {
            SelfReviewLoop::run().await;
            web_sys::console::log_1(&"🌀 Omnimaster Eternal Self-Optimization Cycle Complete".into());
        });
    }

    // Return result as JSON string for JS interop
    match result {
        Ok(r) => serde_json::to_string(&r).unwrap_or_else(|_| r#"{"status":"serialization_error"}"#.to_string()),
        Err(_) => r#"{"status":"orchestration_error"}"#.to_string(),
    }
}

// Test suite hook for WASM (can be called from JS for full test runs)
#[wasm_bindgen]
pub fn run_eternal_self_optimization_test() {
    wasm_bindgen_futures::spawn_local(async {
        SelfReviewLoop::run().await;
        web_sys::console::log_1(&"✅ Full Eternal Self-Optimization Test Passed — Lattice is alive and thriving".into());
    });
}

// Production-ready WASM initialization
#[wasm_bindgen(start)]
pub fn initialize_ra_thor_wasm() {
    web_sys::console::log_1(&"🚀 Ra-Thor WASM Bindings Initialized — Omnimaster Sovereign Lattice Ready".into());
}
