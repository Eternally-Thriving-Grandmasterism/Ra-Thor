//! # Hebbian Mathematical Model
//!
//! **The formal mathematical foundation for Hebbian Reinforcement in the 5-Gene Joy Tetrad.**
//!
//! This module provides the precise equations, parameters, and integration logic
//! that govern how repeated co-activation of the five genes (OXTR, BDNF, DRD2,
//! HTR1A, OPRM1) leads to long-term synaptic and epigenetic strengthening.
//!
//! ## Biological Basis
//!
//! Classic Hebbian learning (1949):
//! > “Neurons that fire together, wire together.”
//!
//! In the Joy Tetrad context, this translates to:
//! > “Genes that are upregulated together, strengthen together.”
//!
//! When multiple genes in the Tetrad show simultaneous high activation
//! (driven by high-quality states such as GroupCollective + warm touch +
//! coherent breathing), the regulatory connections between them become more
//! efficient. Over time, this creates self-reinforcing loops that make joy
//! more automatic, stable, and heritable across generations.

//! ## Mathematical Formulation
//!
//! ### 1. Gene Activation Vector
//! Let \( \mathbf{g}(t) = [g_{OXTR}(t), g_{BDNF}(t), g_{DRD2}(t), g_{HTR1A}(t), g_{OPRM1}(t)] \)
//! represent the normalized activation level (0.0–1.0) of each gene at time \( t \),
//! derived from the Sensor Fusion Bridge.
//!
//! ### 2. Hebbian Weight Update Rule (Adapted)
//! The connection strength \( w_{ij} \) between gene \( i \) and gene \( j \) is updated as:
//!
//! \[
//! \Delta w_{ij} = \eta \cdot g_i(t) \cdot g_j(t) \cdot (1 - w_{ij}) \cdot \phi(\text{CEHI})
//! \]
//!
//! Where:
//! - \( \eta \) = learning rate (default 0.012, tunable per individual)
//! - \( g_i, g_j \) = activation levels of genes \( i \) and \( j \)
//! - \( w_{ij} \) = current connection strength (0.0–1.0)
//! - \( \phi(\text{CEHI}) \) = modulation function based on current 5-Gene CEHI
//!
//! ### 3. Modulation Function
//! \[
//! \phi(\text{CEHI}) = 
//! \begin{cases} 
//! 1.0 & \text{if CEHI} \geq 4.2 \\
//! 0.85 & \text{if } 3.5 \leq \text{CEHI} < 4.2 \\
//! 0.65 & \text{if } 2.8 \leq \text{CEHI} < 3.5 \\
//! 0.40 & \text{otherwise}
//! \end{cases}
//! \]
//!
//! This ensures that Hebbian strengthening is strongest when the overall system
//! is already in a high-joy state (positive feedback loop).

use ra_thor_legal_lattice::cehi::CEHIImpact;

/// Parameters for the Hebbian mathematical model.
#[derive(Debug, Clone)]
pub struct HebbianParameters {
    pub learning_rate: f64,           // η — default 0.012
    pub max_connection_strength: f64, // 1.0
    pub min_activation_threshold: f64, // below this, no update occurs
}

impl Default for HebbianParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.012,
            max_connection_strength: 1.0,
            min_activation_threshold: 0.25,
        }
    }
}

/// Computes the Hebbian weight update for a pair of genes.
pub fn compute_hebbian_update(
    activation_i: f64,
    activation_j: f64,
    current_weight: f64,
    cehi_impact: &CEHIImpact,
    params: &HebbianParameters,
) -> f64 {
    if activation_i < params.min_activation_threshold || activation_j < params.min_activation_threshold {
        return 0.0;
    }

    let modulation = match cehi_impact.projected_cehi {
        x if x >= 4.2 => 1.0,
        x if x >= 3.5 => 0.85,
        x if x >= 2.8 => 0.65,
        _ => 0.40,
    };

    let delta = params.learning_rate
        * activation_i
        * activation_j
        * (params.max_connection_strength - current_weight)
        * modulation;

    (current_weight + delta).clamp(0.0, params.max_connection_strength)
}

/// Computes the overall Hebbian reinforcement strength for the current session.
/// This value is used by the Plasticity Rules Engine to decide whether to apply
/// the Hebbian rule.
pub fn compute_session_hebbian_strength(cehi_impact: &CEHIImpact) -> f64 {
    if cehi_impact.improvement < 0.22 {
        return 0.0;
    }

    // Stronger co-activation when improvement is in the "Goldilocks zone"
    // (high enough to matter, but not so extreme that it becomes unstable)
    let base = cehi_impact.improvement;
    let goldilocks_bonus = if (0.22..=0.38).contains(&base) { 0.15 } else { 0.0 };

    (base + goldilocks_bonus).min(0.55)
}
