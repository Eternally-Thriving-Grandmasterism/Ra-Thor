# Powrush Particle Shaders

## Compute Pipeline Manager

Pipeline cache persistence has been implemented:

- `new(device, initial_cache_data)` accepts optional cache data from disk.
- `get_cache_data()` returns the current cache for saving.
- Call `get_cache_data()` before `destroy()` to persist across runs.

This provides significant startup time improvements.

---
*Phase 1 Consolidation*