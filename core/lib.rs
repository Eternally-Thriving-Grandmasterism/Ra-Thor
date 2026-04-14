// core/lib.rs
// Main entry point for the Ra-Thor Master Sovereign Kernel
// This is the single source of truth that everything in the lattice now calls.

pub mod master_kernel;
pub mod wasm_binding;

// Re-export the master kernel for easy use anywhere in the project
pub use master_kernel::{
    ra_thor_sovereign_master_kernel,
    RequestPayload,
    KernelResult,
};

// Re-export the WASM binding so the browser can call it directly
pub use wasm_binding::ra_thor_master_kernel_js;

// Re-export key subsystems so they can be called cleanly through the kernel
pub mod asre;
pub mod powrush;
pub mod starlink;
pub mod mercyprint;
pub mod mars_colony;

// Convenience wrapper for Rust code (default n = 1_000_000, d = 2)
pub fn ra_thor_call(request: RequestPayload, n: usize) -> KernelResult {
    ra_thor_sovereign_master_kernel(request, n, 2)
}
