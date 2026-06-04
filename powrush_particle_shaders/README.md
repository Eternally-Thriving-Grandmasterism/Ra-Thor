# Powrush Particle Shaders — Hardware Occlusion Queries

## Hardware Occlusion Queries

This module explores traditional hardware occlusion queries and provides guidance on when (and when not) to use them alongside our compute-based culling system.

### Key Takeaways

- **Hardware queries** are powerful for larger bounding volumes but have limitations for fine-grained particle culling.
- **Compute shader culling** (with Hi-Z) is generally preferred for high particle counts due to better performance and easier integration with indirect draws.
- A **hybrid approach** is often optimal: Use compute culling for most particles and hardware queries for important, screen-significant effects.

### Recommended Strategy

Use the `should_use_hardware_query()` helper to decide per-effect which technique to apply.

For most faction events and Resonance Gear visuals, the existing compute + Hi-Z pipeline will deliver better results.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*