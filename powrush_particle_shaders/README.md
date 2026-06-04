# Powrush Particle Shaders — cudaMemPrefetchAsync Optimization

## cudaMemPrefetchAsync Optimization Exploration

This iteration explores `cudaMemPrefetchAsync` as a key optimization technique when using **Unified Memory**.

### What It Does

`cudaMemPrefetchAsync` proactively migrates pages of managed memory to a target device (GPU or CPU) before they are accessed. This reduces expensive on-demand page faults during kernel execution and allows migration to be overlapped with other work using streams.

### Benefits

- Reduces page fault latency inside kernels
- Enables overlapping migration with compute or other transfers
- Improves performance predictability of Unified Memory
- Gives the programmer explicit control over data placement

### When and How to Use It

**Recommended Pattern**:
- Before launching a GPU kernel that will read managed memory, prefetch the data to the GPU.
- After GPU processing, if the CPU will soon read the results, prefetch back to the CPU.
- Use appropriate CUDA streams to allow overlap.

**Example**:
```c
cudaMemPrefetchAsync(particleData, size, gpuDevice, stream);
cudaLaunchKernel(...);  // kernel that accesses particleData
```

### Comparison to Explicit Copies

While `cudaMemPrefetchAsync` is convenient, explicit `cudaMemcpyAsync` often has lower overhead and better performance for large, predictable transfers. Use prefetching when the programming simplicity of Unified Memory is valuable and the migration cost is acceptable.

### Relevance to Powrush

If Unified Memory is used for particle data that occasionally needs CPU access (editing, loading, result inspection), `cudaMemPrefetchAsync` can make GPU-side access much more efficient by migrating data before culling or visibility passes.

It can also be used to bring processed results back to the CPU asynchronously.

### Best Practices

- Prefetch data to the GPU before compute that will access it.
- Prefetch results to the CPU after processing when needed.
- Use streams to overlap migration with other work.
- Avoid prefetching data that won't be used soon.
- Combine with `cudaMemAdvise` for preferred locations and access hints.
- Profile with Nsight Systems to verify reduced page faults and improved kernel performance.

This technique is an important tool for making Unified Memory viable in performance-sensitive parts of the pipeline.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*