# Powrush Particle Shaders — Matrix Multiplication Instruction Sets

## Matrix Multiplication Instruction Sets Exploration

This iteration explores the **low-level matrix multiplication instruction sets** that power modern GPU matrix acceleration.

### Key Instruction Families

**NVIDIA Tensor Cores**:
- `mma.sync` instructions with various shapes and precisions
- Excellent support for mixed-precision (f16, tf32, integers)

**AMD Matrix Cores**:
- MFMA (Matrix Fused Multiply-Add) instructions
- Flexible shapes and data types

**Intel DPAS**:
- Systolic array-based dot product accumulate instructions

### Relationship to Cooperative Matrices

Cooperative matrix extensions provide a higher-level, more portable abstraction over these vendor-specific instructions. When WGSL adds cooperative matrix support, it will map down to the best available hardware instructions on each platform.

### Strategic Value

Understanding these instruction sets helps us anticipate:
- Performance characteristics of future cooperative matrix code
- Precision and throughput trade-offs
- How to best structure workloads when the feature becomes available

This knowledge prepares the Powrush visual system for advanced compute techniques such as neural culling, intelligent LOD, and high-performance procedural effects.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*