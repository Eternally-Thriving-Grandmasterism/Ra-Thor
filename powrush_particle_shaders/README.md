# Powrush Particle Shaders

## Register Pressure Optimization

The WaveLocal Reduction culling shader has been refactored to reduce register pressure:

- Positions loaded as separate scalars (`px`, `py`, `pz`)
- Uses squared distance instead of `distance()`
- `wave_visible_count` moved inside the `lane == 0` branch
- Maintains Structure of Arrays and subgroup operations

These changes help improve occupancy and latency hiding.

---
*Phase 1 Consolidation*