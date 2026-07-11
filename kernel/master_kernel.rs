/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.4 (CUDA Kernel Path Exposed)  
**Date**: 2026-07-11

## CUDA Support
Real CUDA kernel from `kernel/cuda/tolc_compute_kernel.cu` is now fully wired.

New methods:
- `tick_cuda_batch()`
- `tick_with_priority_queue_cuda()`

These launch the actual CUDA kernel for maximum parallel throughput on NVIDIA GPUs.
*/

use crate::kernel::tolc_proof_carrying::{
    conduct_deliberation_batch_cuda,
    allocation_priority_queue_cuda,
    // ... other imports
};

impl MasterKernel {
    // ... existing methods ...

    /// Real CUDA batch deliberation
    pub fn tick_cuda_batch(&mut self, candidate_actions: &[String]) -> Vec<(String, f64, f64)> {
        self.tick_count += 1;
        conduct_deliberation_batch_cuda(
            candidate_actions,
            &self.current_state,
            &self.weights,
            &self.utf_thresholds,
        )
    }

    /// CUDA batch + sorted priority queue (highest performance path)
    pub fn tick_with_priority_queue_cuda(&mut self, candidate_actions: &[String]) -> Vec<(String, f64, f64)> {
        self.tick_count += 1;
        allocation_priority_queue_cuda(
            candidate_actions,
            &self.current_state,
            &self.weights,
            &self.utf_thresholds,
        )
    }
}

/*!
## All Tick Variants Now Available

| Method                        | Backend     | Parallel | Use Case                     |
|-------------------------------|-------------|----------|------------------------------|
| tick()                        | CPU single  | No       | Simple decisions             |
| tick_with_priority_queue()    | CPU queue   | No       | Ranked list (CPU)            |
| tick_gpu_batch()              | Rayon       | Yes      | Good parallel (no CUDA)      |
| tick_with_priority_queue_gpu()| Rayon       | Yes      | Ranked parallel (no CUDA)    |
| tick_cuda_batch()             | Real CUDA   | Yes      | Maximum throughput (NVIDIA)  |
| tick_with_priority_queue_cuda()| Real CUDA | Yes      | Best performance path        |
*/
