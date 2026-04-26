//! # Hebbian Reinforcement Module
//!
//! **The fourth and most granular plasticity rule in the Ra-Thor Plasticity Engine v2.**
//!
//! This module implements **Hebbian Reinforcement** — the biological principle that
//! “neurons that fire together, wire together” — applied to the 5-Gene Joy Tetrad.
//!
//! ## Biological Foundation
//!
//! When multiple genes in the Joy Tetrad (OXTR, BDNF, DRD2, HTR1A, OPRM1) are
//! upregulated together during high-quality states (e.g., GroupCollective laughter +
//! warm touch + coherent breathing), the regulatory pathways between them become
//! stronger and more efficient over time.
//!
//! This creates **self-reinforcing loops** that make the entire system:
//! - More sensitive to joy-inducing stimuli
//! - More resistant to stress and regression
//! - More automatic and effortless across generations
//!
//! ## Role in the 200-Year+ Mercy Legacy
//!
//! Hebbian Reinforcement is the mechanism that turns repeated high-quality practice
//! into **permanent neural and epigenetic architecture**. By F4 (2226), it helps ensure
//! that children are born with nervous systems already “pre-wired” for bonding,
//! motivation, stability, and ecstasy — because the gene regulatory networks have been
//! strengthened across generations through repeated co-activation.

use ra_thor_legal_lattice::cehi::CEHIImpact;

/// Hebbian Reinforcement rule — strengthens co-activation patterns between genes.
pub struct HebbianReinforcement;

impl HebbianReinforcement {
    /// Creates a new Hebbian Reinforcement instance.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates whether Hebbian Reinforcement should be applied.
    ///
    /// Triggered when multiple genes show strong simultaneous improvement,
    /// indicating high-quality co-activation states.
    pub async fn evaluate(
        &self,
        impact: &CEHIImpact,
    ) -> Result<HebbianResult, crate::PlasticityError> {
        // Hebbian reinforcement is triggered on high-quality, multi-gene days
        // (when overall improvement is strong and consistent)
        let should_apply = impact.improvement >= 0.22 && impact.improvement <= 0.45;

        if should_apply {
            Ok(HebbianResult {
                rule_name: "HebbianReinforcement".to_string(),
                should_apply: true,
                strength: impact.improvement * 0.75, // Moderate but sustained effect
                description: "Strengthening co-activation patterns between Joy Tetrad genes".to_string(),
            })
        } else {
            Ok(HebbianResult {
                rule_name: "HebbianReinforcement".to_string(),
                should_apply: false,
                strength: 0.0,
                description: "Co-activation below threshold for meaningful Hebbian strengthening".to_string(),
            })
        }
    }
}

/// Result of a Hebbian Reinforcement evaluation.
#[derive(Debug, Clone)]
pub struct HebbianResult {
    pub rule_name: String,
    pub should_apply: bool,
    pub strength: f64,
    pub description: String,
}
