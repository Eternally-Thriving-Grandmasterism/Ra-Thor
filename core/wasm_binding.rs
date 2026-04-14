use wasm_bindgen::prelude::*;
use crate::master_kernel::*;

#[wasm_bindgen]
pub fn ra_thor_master_kernel_js(
    request_json: &str,
    n: usize,
    d: u32,
) -> String {
    let request: RequestPayload = serde_json::from_str(request_json).unwrap_or_default();
    let result = ra_thor_sovereign_master_kernel(request, n, d);
    serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
}
