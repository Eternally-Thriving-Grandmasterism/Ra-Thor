# Powrush Particle Shaders

## Register Pressure Optimization

Updated `ComputeCullingParams` to use `max_cull_distance_squared` to match the optimized shader.

When preparing parameters on the host, compute:
```rust
let max_dist_squared = max_distance * max_distance;
```

---
*Phase 1 Consolidation*