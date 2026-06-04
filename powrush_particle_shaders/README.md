# Powrush Particle Shaders

## Compute Pipeline Manager

Automatic pipeline cache persistence is now supported.

### Usage

```rust
let cache_path = Some(std::path::PathBuf::from("pipeline_cache.bin"));
let mut manager = ComputePipelineManager::new(device, cache_path);

// ... use pipelines ...

manager.destroy(); // Cache is automatically saved
```

The cache is also saved automatically if the manager is dropped.

---
*Phase 1 Consolidation*