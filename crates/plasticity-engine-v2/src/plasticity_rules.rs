//! # Plasticity Rules Engine
//!
//! **Defines and evaluates the four core plasticity rules for the 5-Gene Joy Tetrad.**
//!
//! This module contains the decision logic that determines whether a plasticity update
//! should be applied based on the calculated CEHI improvement from the Epigenetic Updater.
//!
//! ## The Four Core Rules
//!
//! 1. **JoyTetradLock** — Triggered on exceptional days (top-tier improvements).  
//!    Represents the state where all five genes are strongly reinforcing each other.
//!
//! 2. **MetaplasticReinforcement** — Triggered on strong, consistent days.  
//!    Encourages deeper, longer-lasting epigenetic changes through repeated activation.
//!
//! 3. **HomeostaticMaintenance** — Triggered on good baseline days.  
//!    Keeps the system stable and prevents regression while still providing gentle progress.
//!
//! 4. **HebbianReinforcement** (future expansion) — Will handle fine-grained, synapse-level updates.
//!
//! ## Integration
//!
//! This engine works in tight coordination with:
//! - `EpigeneticUpdater` (sibling module)
//! - `ra-thor-legal-lattice` (for Mercy Legacy Fund tier decisions)
//! - The 7 Living Mercy Gates (updates are only applied when all gates pass)
//!
//! ## Design Principle
//!
//! The rules follow the TOLC Mercy Compiler: **Truth** (data-driven thresholds),  
//! **Order** (clear, auditable decision tree), **Love** (encourages consistent practice),  
//! and **Clarity** (transparent, explainable outcomes).

use ra_thor_legal_lattice::cehi::CEHIImpact;

/// The Plasticity Rules Engine — decides whether and how strongly to apply updates.
pub struct PlasticityRulesEngine;

impl PlasticityRulesEngine {
    /// Creates a new Plasticity Rules Engine instance.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the CEHI improvement and returns the appropriate plasticity rule.
    ///
    /// Thresholds are aligned with the revised tier logic in the Epigenetic Updater
    /// and calibrated to accelerate the 200-year global mercy legacy.
    pub async fn evaluate(
        &self,
        impact: &CEHIImpact,
    ) -> Result<RuleResult, crate::PlasticityError> {
        if impact.improvement >= 0.32 {
            Ok(RuleResult {
                rule_name: "JoyTetradLock".to_string(),
                should_apply: true,
                strength: impact.improvement,
            })
        } else if impact.improvement >= 0.18 {
            Ok(RuleResult {
                rule_name: "MetaplasticReinforcement".to_string(),
                should_apply: true,
                strength: impact.improvement * 0.85,
            })
        } else if impact.improvement >= 0.12 {
            Ok(RuleResult {
                rule_name: "HomeostaticMaintenance".to_string(),
                should_apply: true,
                strength: impact.improvement * 0.65,
            })
        } else {
            Ok(RuleResult {
                rule_name: "BelowThreshold".to_string(),
                should_apply: false,
                strength: 0.0,
            })
        }
    }
}

/// Represents the result of a plasticity rule evaluation.
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// Name of the rule that was selected
    pub rule_name: String,
    /// Whether the update should be applied
    pub should_apply: bool,
    /// Strength multiplier for the update (affects consolidation speed)
    pub strength: f64,
}
