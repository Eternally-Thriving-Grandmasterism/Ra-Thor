/*!
# GPU Compute Pipeline — Placeholder Cleanup (v0.4)

- compute_tu_on_gpu_or_cpu now has clearer comment that it is a stub for real WGPU/CUDA dispatch
- All major TODOs for real kernel integration are now documented as future work (not blockers)
*/

fn compute_tu_on_gpu_or_cpu(...) -> f64 {
    // Placeholder for real GPU kernel (WGPU or CUDA dispatch to tolc_compute_kernel.cu)
    // In production this would transfer features to GPU, launch kernel, and read back results
    0.6 + (action.len() as f64 % 5) * 0.05
}
