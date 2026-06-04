# Powrush Particle Shaders — Memory Fence Performance Costs

## Memory Fence Performance Costs Exploration

This iteration explores the **performance costs** of memory fences and how they vary by scope and usage.

### What Makes Fences Expensive?

Fences can cause:
- Execution stalls while waiting for memory visibility
- Cache flushes or invalidations
- Prevention of instruction reordering (hurting latency hiding)
- Increased memory traffic due to visibility requirements

### Cost by Scope

**Subgroup Scope**:
- Lowest cost.
- Lightweight wave synchronization with minimal cache impact.
- Preferred for performance-critical code.

**Workgroup Scope**:
- Moderate to high cost.
- Requires cross-workgroup synchronization and stronger visibility.
- Noticeable performance impact if overused.

**Device / QueueFamily Scope**:
- Highest cost.
- Should be avoided in real-time rendering and compute pipelines.

### Impact on Powrush Architecture

Our strong preference for **Subgroup-scoped** operations and wave-local techniques (ballot, shuffle, wave-local reduction) keeps memory fence costs low in the particle culling and visibility system.

Using stronger fences than necessary would introduce avoidable overhead.

### Best Practices

- Default to Subgroup scope.
- Use stronger fences only when required by the algorithm (e.g., Workgroup-scoped cooperative matrices).
- Minimize fence frequency through wave-local work aggregation.
- Profile to detect unnecessary fence overhead.

This reinforces why our wave-centric design is performance-friendly.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*