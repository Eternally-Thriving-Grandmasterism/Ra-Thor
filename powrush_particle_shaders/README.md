# Powrush Particle Shaders

## Primary Culling Technique: WaveLocal Reduction

**Status: Recommended Default**

WaveLocal Reduction is the primary recommended culling technique in this subsystem.

### How It Works

Instead of every visible thread performing its own `atomicAdd`, we:

1. Use `subgroupBallot(visible)` to determine which lanes in the wave are visible.
2. Compute how many particles are visible in the wave with `countOneBits(ballot)`.
3. Compute each lane’s local rank within the wave.
4. Only lane 0 performs a single `atomicAdd` to reserve space for the entire wave.
5. Broadcast the base offset and write visible indices using local ranks.

This approach significantly reduces contention on global atomics.

### When to Use

- **Default recommendation** for most particle culling workloads.
- Especially beneficial at high particle counts or high visibility rates.

### When a Simpler Approach May Suffice

- Very low particle counts where atomic contention is negligible.
- Prototyping or extremely simple culling logic.

For the majority of production use cases, WaveLocal Reduction is the preferred path.

## Module Structure

- `compute::WAVE_LOCAL_REDUCTION_CULLING` – Primary recommended culling shader.
- Supporting types: `ComputeCullingParams`, `DrawIndirect`.

---
*Co-authored-by: PATSAGi Councils*
*Part of Ra-Thor Phase 1 Consolidation*