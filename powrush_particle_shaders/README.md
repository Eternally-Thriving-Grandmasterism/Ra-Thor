# Powrush Particle Shaders — GPU Branch Prediction

## GPU Branch Prediction Investigation

This iteration investigates how **GPUs handle branching** and why it differs significantly from CPUs.

### GPU vs CPU Approach

CPUs use sophisticated branch predictors with history to speculate on branch outcomes and hide latency.

GPUs prioritize massive parallelism and latency hiding across many threads instead of deep speculation on individual control flow. When a warp hits a branch:
- If all threads take the same path → efficient execution.
- If threads diverge → the warp serializes both paths.

GPUs have much more limited dynamic branch prediction. The emphasis is on **warp uniformity** rather than prediction.

### Implications for Shader Writing

- The most effective strategy is to **minimize divergence** (keep warps uniform when possible).
- Wave-uniform operations like `subgroupBallot` and shuffle are powerful because they execute identically across the wave.
- Structuring code so that divergent work happens before or after wave-uniform phases helps overall efficiency.

### Relevance to Powrush

Visibility tests in culling introduce potential divergence. Our **WaveLocal Reduction** technique (ballot + shuffle) helps by performing the divergent visibility test first, then moving into a more uniform compaction phase. This structure reduces the performance penalty of divergence.

Understanding the limited role of branch prediction on GPUs reinforces why our wave-local, uniformity-focused techniques are effective.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*