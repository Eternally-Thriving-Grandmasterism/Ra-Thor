// core/lib.rs
// Main entry point for the Ra-Thor Master Sovereign Kernel

pub mod master_kernel;
pub mod wasm_binding;
pub mod global_cache;   // ← NEW: Global Cache Module

// Re-export everything
pub use master_kernel::{ra_thor_sovereign_master_kernel, RequestPayload, KernelResult};
pub use wasm_binding::ra_thor_master_kernel_js;
pub use global_cache::GlobalCache;

// Re-export key subsystems
pub mod asre;
pub mod powrush;
pub mod starlink;
pub mod mercyprint;
pub mod mars_colony;

// Convenience wrapper
pub fn ra_thor_call(request: RequestPayload, n: usize) -> KernelResult {
    ra_thor_sovereign_master_kernel(request, n, 2)
}
