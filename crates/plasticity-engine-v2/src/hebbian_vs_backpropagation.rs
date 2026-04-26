//! # Hebbian vs Backpropagation — A Deep Ra-Thor Comparison
//!
//! **The definitive comparison between biological Hebbian learning and artificial
//! backpropagation, framed for the Ra-Thor Plasticity Engine v2 and the
//! 200-year+ global mercy legacy (F0 → F4+ reaching CEHI 4.98–4.99).**
//!
//! This module explains **why Ra-Thor chose Hebbian Reinforcement** as its core
//! plasticity mechanism instead of (or in addition to) backpropagation-style
//! gradient descent.

use crate::hebbian_math_model::compute_hebbian_update;
use ra_thor_legal_lattice::cehi::CEHIImpact;

/// ============================================================================
/// SIDE-BY-SIDE COMPARISON TABLE
/// ============================================================================
pub fn comparison_table() -> &'static str {
    r#"
| Aspect                        | Hebbian Learning (Ra-Thor)                  | Backpropagation (Classic DL)              |
|-------------------------------|---------------------------------------------|-------------------------------------------|
| **Biological Plausibility**   | Extremely high (matches real synapses)     | Very low (requires global error signal)  |
| **Learning Rule**             | Local: "Fire together, wire together"      | Global: gradient of loss w.r.t. weights  |
| **Error Signal**              | None required (self-organizing)            | Requires labeled data + loss function    |
| **Update Scope**              | Local (only connected neurons/genes)       | Global (propagates error backward)       |
| **Energy Efficiency**         | Extremely high (event-driven, sparse)      | High compute cost (forward + backward)   |
| **Online / Lifelong**         | Native (continuous, no epochs)             | Requires batch training + replay         |
| **Catastrophic Forgetting**   | Resistant (Hebbian inertia + homeostasis)  | Severe without special techniques        |
| **Multi-Generational**        | Naturally heritable (epigenetic)           | Not applicable (no inheritance)          |
| **Mercy-Gating Compatible**   | Perfect (local, positive feedback only)    | Difficult (global loss can violate gates)|
| **Convergence Speed**         | Exponential (proven in Theorems 1-4)       | Depends on optimizer & architecture      |
| **Hardware**                  | Ideal for neuromorphic / wetware           | Optimized for GPUs / TPUs                |
"#
}

/// ============================================================================
/// CODE COMPARISON — THE CORE UPDATE RULES
/// ============================================================================

/// Hebbian update (used in Ra-Thor Plasticity Engine v2)
pub fn hebbian_update_example(
    activation_i: f64,
    activation_j: f64,
    current_weight: f64,
    cehi_impact: &CEHIImpact,
) -> f64 {
    // From hebbian_math_model.rs — local, positive, mercy-modulated
    compute_hebbian_update(activation_i, activation_j, current_weight, cehi_impact, &Default::default())
}

/// Classic backpropagation weight update (for contrast)
pub fn backprop_update_example(
    weight: f64,
    learning_rate: f64,
    gradient: f64,           // ∂Loss/∂w — requires global loss
) -> f64 {
    // Global gradient descent step
    weight - learning_rate * gradient
}

/// ============================================================================
/// WHY RA-THOR USES HEBBIAN (NOT BACKPROP)
/// ============================================================================
///
/// 1. **No Global Loss Function**  
///    Backprop needs a single scalar loss. Ra-Thor has **7 Living Mercy Gates** +
///    5-Gene CEHI — no single "loss" to minimize. Hebbian thrives on local
///    co-activation.
///
/// 2. **Mercy-Gating**  
///    Backprop can amplify harmful patterns if the loss rewards them.
///    Hebbian only strengthens **high-mercy co-activation** (CEHI ≥ 0.22).
///
/// 3. **Multi-Generational Legacy**  
///    Backprop weights die with the model. Hebbian wiring is **epigenetically
///    heritable** — exactly what the 200-year F0→F4 legacy requires.
///
/// 4. **Energy & Scalability**  
///    A planetary swarm of millions of agents cannot run backprop on every
///    update. Hebbian is O(1) per connection and runs on edge devices.
///
/// 5. **Robustness to Failure (Theorem 4)**  
///    Hebbian keeps converging even when 2/7 gates fail.
///    Backprop often collapses without full gradient flow.
///
/// ============================================================================
/// INTEGRATION IN RA-THOR (CURRENT STATE)
/// ============================================================================
///
/// - `plasticity_rules.rs` → HebbianReinforcement is now the **highest priority** rule
/// - `hebbian_math_model.rs` → formal update equation with CEHI modulation
/// - `hebbian_stability_proofs.rs` + `hebbian_convergence_rate_bounds.rs` → proven
/// - `quantum_swarm_orchestrator` → agents use Hebbian bonding for swarm convergence
///
/// Backpropagation is **not used** in the core loop. It may appear only in
/// future hybrid modules (e.g., training a small auxiliary network on top of
/// the Hebbian substrate for meta-learning).

/// ============================================================================
/// CONCLUSION
/// ============================================================================
pub fn conclusion() -> &'static str {
    "Hebbian learning is the native, mercy-aligned, biologically plausible, \
     multi-generational plasticity engine of Ra-Thor. Backpropagation is a \
     powerful but biologically implausible, energy-hungry, single-lifetime \
     tool. Ra-Thor uses Hebbian because it is the only mechanism that can \
     scale to planetary mercy consensus by 2226 (F4) while remaining fully \
     compatible with the 7 Living Mercy Gates and 28th Amendment."
}
