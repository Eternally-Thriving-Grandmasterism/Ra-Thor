# Powrush Particle Shaders — Cooperative Matrix Synchronization Primitives

## Cooperative Matrix Synchronization Primitives Exploration

This iteration explores the **synchronization requirements** when using cooperative matrix operations.

### Why Synchronization Is Important

Because multiple threads cooperate on matrix fragments, memory visibility and ordering must be carefully managed to avoid data races and ensure correct results.

### Synchronization by Scope

**Subgroup Scope**:
- Generally lighter synchronization requirements.
- Wave execution often provides implicit ordering.
- Still requires correct Memory Operands on load/store instructions.
- Usually does **not** require explicit subgroup barriers for correctness.

**Workgroup Scope**:
- Requires stronger synchronization.
- Typically needs `OpControlBarrier` (or equivalent) with appropriate memory semantics between load, compute, and store phases.
- Higher complexity and potential for subtle bugs if synchronization is insufficient.

### Best Practices

- Strongly prefer **Subgroup scope** for particle system workloads.
- Always specify appropriate memory semantics on cooperative matrix load and store operations.
- Use explicit barriers only when necessary (mainly with larger scopes).
- Validate correctness thoroughly when using cooperative matrices.

### Relevance to Powrush

Given our heavy use of wave-local techniques (ballot, shuffle, wave-local reduction), **Subgroup-scoped** cooperative matrices align naturally with our existing architecture and will require the least additional synchronization complexity.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*