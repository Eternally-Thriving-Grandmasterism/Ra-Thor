# Powrush Particle Shaders — Memory Operand Details

## Memory Operand Details Exploration

This iteration provides a detailed look at the **Memory Operands** used with cooperative matrix load and store operations.

### Purpose of Memory Operands

Memory Operands control ordering, visibility, and coherence when loading or storing cooperative matrices. They are essential for correctness in multi-threaded execution.

### Key Operands

- `MakePointerAvailable` / `MakePointerAvailableKHR`
  - Ensures that data written by a store becomes available to other threads/scopes.

- `MakePointerVisible` / `MakePointerVisibleKHR`
  - Ensures that data is visible to the loading thread/scope before a load occurs.

- `NonPrivatePointer` / `NonPrivatePointerKHR`
  - Indicates the memory may be accessed by multiple threads (almost always required for cooperative matrices).

- `Volatile`, `Aligned`, and others control caching and alignment behavior.

### Scope-Dependent Usage

**Subgroup Scope**:
- `NonPrivatePointer` is typically required.
- Visibility operands (`MakePointerAvailable` / `MakePointerVisible`) are used as needed when mixing with other memory operations.

**Workgroup Scope**:
- Stronger Acquire/Release semantics are often combined with explicit barriers.

### Guidance for Powrush

For recommended **Subgroup-scoped** cooperative matrix usage:
- Include `NonPrivatePointer`.
- Apply visibility operands conservatively when combining cooperative matrix operations with regular loads/stores.
- Validate correctness early.

These details ensure safe and correct use of cooperative matrices once WGSL support becomes available.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*