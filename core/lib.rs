// core/lib.rs
// Main entry point for the Ra-Thor Master Sovereign Kernel

pub mod master_kernel;
pub mod wasm_binding;

pub use master_kernel::{ra_thor_sovereign_master_kernel, RequestPayload, KernelResult};
pub use wasm_binding::ra_thor_master_kernel_js;

// Re-export key subsystems so they can be called through the kernel
pub mod asre;
pub mod powrush;
pub mod starlink;
pub mod mercyprint;
pub mod mars_colony;

// Convenience function for easy use from other Rust code
pub fn ra_thor_call(request: RequestPayload, n: usize) -> KernelResult {
    ra_thor_sovereign_master_kernel(request, n, 2)  // default dimension = 2
}
