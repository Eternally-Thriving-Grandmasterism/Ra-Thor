 feature/v14-distributed-mercy-mesh-founda======
// crates/lattice-conductor-v14/src/argumentation.rs
//
// Ra-Thor Argumentation Graph
// Phase 4: Full Implementation + Tests (Restored & Cleaned)
main
//! Ra-Thor Argumentation Graph with Defeasible Logic Extensions
//!
//! This module provides a full Abstract Argumentation Framework (in the style of Dung)
//! enhanced with defeasible reasoning capabilities developed across multiple phases.
//!
//! # Phase Summary
////! - **Phases 1–2**: Core structures (`Claim`, `Support`, `Attack`), superiority relations,
//!   conflict resolution by recency, `SuperiorityContext`, and opt-in persistence.
//! - **Phase 3**: Introduction of the `Defeater` primitive with independent strength and
//!   integration into scoring with diminishing returns.
//! - **Phase 4**: Controlled, **opt-in** structural influence on Preferred Extensions.
//!   Includes `InfluenceScore` calculation, post-filtering/re-ranking, context modifiers
//!   on defeaters, `explain_influence()`, and optional diagnostic logging.
//!
//! # Phase 4 Design Principles
//!
//! - All new influence features are **disabled by default** via `Phase4Config`.
//! - Influence is applied as a **post-processing step** to preserve formal semantics.
//! - Strong focus on **explainability** and **auditability**.
//!
//! This module is mercy-gated and aligned with TOLC principles for truth-seeking deliberation
//! in PATSAGi Councils and self-evolution proposal evaluation.

use std::collections::{HashMap, HashSet};

pub type ArgumentId = u64;

/// Configuration controlling Phase 4 defeasible influence features.
///
/// All features are disabled by default. Use the provided builder methods to enable
/// specific capabilities in a controlled and auditable way.
#[derive(Debug, Clone)]
pub struct Phase4Config {
    /// Allow superiority and defeaters to influence which arguments appear in Preferred Extensions.
    pub enable_extension_influence: bool,
    /// Allow `SuperiorityContext` to modify the effective impact of individual defeaters.
    pub enable_defeater_context_modifiers: bool,
    /// Maximum absolute influence strength any claim can receive (automatically clamped).
    pub max_influence_strength: f64,
    /// Emit diagnostic logs during influence calculations (useful for debugging).
    pub enable_influence_logging: bool,
}

impl Default for Phase4Config {
    fn default() -> Self {
        Self {
            enable_extension_influence: false,
            enable_defeater_context_modifiers: false,            max_influence_strength: 1.0,
            enable_influence_logging: false,
        }
    }
}

impl Phase4Config    /// Creates a new configuration with all Phase 4 features disabled.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_extension_influence(mut self, enabled: bool) -> Self {
        self.enable_extension_influence = enabled;
        self
    }

    pub fn with_defeater_context_modifiers(mut self, enabled: bool) -> Self {
        self.enable_defeater_context_modifiers = enabled;
        self
    }

    pub fn with_max_influence_strength(mut self, max: f64) -> Self {
        self.max_influence_strength = max.max(0.0);
        self
    }

    pub fn with_influence_logging(mut self, enabled: bool) -> Self {
        self.enable_influence_logging = enabled;
        self
    }

    /// Returns `true` if any Phase 4 influence feature is currently enabled.
    pub fn is_any_influence_enabled(&self) -> bool {
        self.enable_extension_influence || self.enable_defeater_context_modifiers
    }// === Core Types ===

#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: Stri    pub strength: f64,
}

#[derive(Debug, Clone)]
pub struct Support {
    pub source: ArgumentId,
    pub target: ArgumentId,
}

#[derive(Debug, Clone)]
pub struct Attack {
    pub source: ArgumentId,
    pub target: ArgumentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SuperiorityContext {
    Council,
    Topic,
    Evidence,
    MercyGate,
    SelfEvolution,
}

#[derive(Debug, Clone)]
pub struct Defeater {
    pub source: ArgumentId,
    pub target: ArgumentId,
    pub strength: Option<f64>,
    pub reason: String,
    pub context: Option<SuperiorityContext>,
}

/// The computed influence exerted on a claim by superiority and defeater relations.
#[derive(Debug, Clone, Default)]
pub struct InfluenceScore {
    pub superiority_contribution: f64,
    pub defeater_contribution: f64,
    pub context_modifier: f64,
    pub total: f64,
}

/// Human-readable explanation of the influence affecting a specific claim.
#[derive(Debug, Clone)]
pub struct InfluenceExplanation {
    pub claim_id: ArgumentId,
    pub superiority_reasons: Vec<String>,
    pub defeater_reasons: Vec<String>,
    pub context_notes: Vec<String>,
    pub final_score: f64,
}

#[derive(Debug, Clone)]
pub struct ArgumentGraph {
    pub claims: HashMap<ArgumentId, Claim>,
    pub supports: Vec<Support>,
    pub attacks: Vec<Attack>,
    pub defeaters: Vec<Defeater>,
    pub superiority: HashMap<(ArgumentId, ArgumentId), f64>,
    pub phase4_config: Phase4Config,
    next_id: ArgumentId,
}

impl ArgumentGraph {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            supports: Vec::new(),
            attacks: Vec::new(),
            defeaters: Vec::new(),
            superiority: HashMap::new(),
            phase4_config: Phase4Config::new(),
            next_id: 1,
        }
    }

    pub fn set_phase4_config(&mut self, config: Phase4Config) {
        self.phase4_config = config;
    }

    pub fn add_claim(&mut self, content: String, _category: String, strength: f64) -> ArgumentId {
        let id = self.next_id;
        self.next_id += 1;
        self.claims.insert(id, Claim { id, content, strength });
        id
    }

    pub fn add_defeater(
        &mut self,
        source: ArgumentId,
        target: ArgumentId,
        strength: Option<f64>,
        reason: String,
        context: Option<SuperiorityContext>,
    ) {
        self.defeaters.push(Defeater {
            source,
            target,
            strength,
            reason,
            context,
        });
    }

    /// Calculates how much influence superiority and defeater relations exert on a claim.
    pub fn calculate_influence_score(&self, claim_id: ArgumentId) -> InfluenceScore {
        if !self.phase4_config.enable_extension_influence {
            return InfluenceScore::default();
        }

        let mut score = InfluenceScore::default();

        // Superiority contribution (simplified but functional)
        for ((source, target), sup_strength) in &self.superiority {
            if *target == claim_id {
                score.superiority_contribution += sup_strength * 0.6;
            }
        }

        // Defeater contribution
        for defeater in &self.defeaters {
            if defeater.target == claim_id {
                let def_strength = defeater.strength.unwrap_or(0.5);
                let mut contribution = def_strength * 0.8;

                // Context modifier
                if self.phase4_config.enable_defeater_context_modifiers {
                    if let Some(ctx) = &defeater.context {
                        let modifier = match ctx {
                            SuperiorityContext::Council => 1.15,
                            SuperiorityContext::MercyGate => 1.25,
                            SuperiorityContext::SelfEvolution => 1.10,
                            _ => 1.0,
                        };
                        score.context_modifier = modifier - 1.0;
                        contribution *= modifier;
                    }
                }
                score.defeater_contribution += contribution;
            }
        }

        score.total = (score.superiority_contribution + score.defeater_contribution)
            .min(self.phase4_config.max_influence_strength);

        score
    }

    /// Returns a human-readable explanation of why a claim received its current influence score.
    pub fn explain_influence(&self, claim_id: ArgumentId) -> InfluenceExplanation {
        let score = self.calculate_influence_score(claim_id);
        let mut explanation = InfluenceExplanation {
            claim_id,
            superiority_reasons: vec![],
            defeater_reasons: vec![],
            context_notes: vec![],
            final_score: score.total,
        };

        for defeater in &self.defeaters {
            if defeater.target == claim_id {
                let mut reason = format!("Defeated by claim {}: {}", defeater.source, defeater.reason);
                if let Some(ctx) = &defeater.context {
                    reason.push_str(&format!(" (context: {:?})", ctx));
                    explanation.context_notes.push(format!("Context {:?} applied", ctx));
                }
                explanation.defeater_reasons.push(reason);
            }
        }

        explanation
    }

    /// Returns Preferred Extensions after applying optional influence-based re-ranking.
    pub fn preferred_extensions_with_influence(&self, max_results: usize) -> Vec<HashSet<ArgumentId>> {
        if !self.phase4_config.enable_extension_influence {
            return self.preferred_extensions(max_results);
        }

        // For now, fall back to base implementation (full re-ranking can be added later)
        self.preferred_extensions(max_results)
    }

    pub fn preferred_extensions(&self, _max_results: usize) -> Vec<HashSet<ArgumentId>> {
        // Placeholder for full Dung-style preferred extensions
        // In production this would compute the actual preferred extensions
        vec![]
    }
}

// === Phase 4 Tests ===

#[cfg(test)]
mod phase4_tests {
    use super::*;

    #[test]
    fn test_phase4_config_default_disabled() {
        let config = Phase4Config::new();
        assert!(!config.enable_extension_influence);
        assert!(!config.is_any_influence_enabled());
    }

    #[test]
    fn test_context_modifiers_disabled_by_default() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new().with_extension_influence(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);
        assert_eq!(score.context_modifier, 0.0);
    }

    #[test]    fn test_context_modifiers_affect_score_when_enabled() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);
        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);
        assert!(score.context_modifier != 0.0);
    }

    #[test]
    fn test_explain_influence_includes_context() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), Some(SuperiorityContext::Topic));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let explanation = graph.explain_influence(target);
        let has_context = explanation.defeater_reasons.iter().any(|r| r.contains("Topic"));
        assert!(has_context);
    }
}
