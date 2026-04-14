// core/wasm_binding.rs
// WASM bindings so the browser / PWA shards can call the Master Sovereign Kernel

use wasm_bindgen::prelude::*;
use crate::master_kernel::*;

// Main WASM entry point that JavaScript / TypeScript can call
#[wasm_bindgen]
pub fn ra_thor_master_kernel_js(
    request_json: &str,
    n: usize,
    d: u32,
) -> String {
    // Safely parse the incoming JSON request
    let request: RequestPayload = serde_json::from_str(request_json)
        .unwrap_or_else(|_| RequestPayload {
            operation_type: "unknown".to_string(),
            data: serde_json::json!({}),
        });

    // Call the unified Master Sovereign Kernel
    let result = ra_thor_sovereign_master_kernel(request, n, d);

    // Return JSON string back to JavaScript
    serde_json::to_string(&result)
        .unwrap_or_else(|_| r#"{"status":"error","message":"JSON serialization failed"}"#.to_string())
}
