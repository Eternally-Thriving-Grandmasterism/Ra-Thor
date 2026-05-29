//! Ra-Thor Argumentation Graph with Defeasible Logic Extensions
//!
//! This module provides a full Abstract Argumentation Framework (in the style of Dung)
//! enhanced with defeasible reasoning capabilities developed across multiple phases.
//!
//! # Phase Summary
//!
//! - **Phases 1–2**: Core structures, superiority relations, conflict resolution,
//!   `SuperiorityContext`, and opt-in persistence.
//! - **Phase 3**: `Defeater` primitive with independent strength and scoring integration.
//! - **Phase 4**: Controlled, opt-in structural influence on Preferred Extensions via
//!   `InfluenceScore`, post-filtering/re-ranking, context modifiers, explainability,
//!   and optional logging.
//!
//! # Phase 4 Design Principles
//!
//! - All influence features are **disabled by default** via `Phase4Config`.
//! - Influence is applied as **post-processing** to preserve formal semantics.
//! - Strong emphasis on **explainability**, **auditability**, and mercy-aligned design.

use std::collections::{HashMap, HashSet};

pub type ArgumentId = u64;

/// Configuration for Phase 4 defeasible influence features.
///
/// All features are disabled by default for safety and backward compatibility.
#[derive(Debug, Clone)]
pub struct Phase4Config {
    pub enable_extension_influence: bool,
    pub enable_defeater_context_modifiers: bool,
    pub max_influence_strength: f64,
    pub enable_influence_logging: bool,
}

impl Default for Phase4Config {
    fn default() -> Self {
        Self {
            enable_extension_influence: false,
            enable_defeater_context_modifiers: false,
            max_influence_strength: 0.3,
            enable_influence_logging: false,
        }
    }
}

impl Phase4Config {
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

    pub fn with_max_influence_strength(mut self, strength: f64) -> Self {
        self.max_influence_strength = strength.clamp(0.0, 1.0);
        self
    }

    pub fn with_influence_logging(mut self, enabled: bool) -> Self {
        self.enable_influence_logging = enabled;
        self
    }

    pub fn is_any_influence_enabled(&self) -> bool {
        self.enable_extension_influence || self.enable_defeater_context_modifiers
    }
}

#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String,
    pub strength: f64,
    pub is_strict: bool,
}

#[derive(Debug, Clone)]
pub struct Support {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

#[derive(Debug, Clone)]
pub struct Attack {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// Context type for superiority and defeater relations.
///
/// Includes standard contexts plus mercy-aligned (`MercyGate`) and evolution-oriented (`SelfEvolution`) variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperiorityContext {
    Council,
    Topic,
    General,
    MercyGate,
    SelfEvolution,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct Superiority {
    pub stronger: ArgumentId,
    pub weaker: ArgumentId,
    pub context: Option<SuperiorityContext>,
}

/// A defeater weakens a target claim without fully invalidating it.
///
/// Carries an optional human-readable `reason` for improved explainability.
#[derive(Debug, Clone)]
pub struct Defeater {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub strength: f64,
    pub provided_by: String,
    pub reason: String,
    pub context: Option<SuperiorityContext>,
}

/// Computed influence on a claim from superiority and defeater relations.
#[derive(Debug, Clone, Default)]
pub struct InfluenceScore {
    pub superiority_contribution: f64,
    pub defeater_contribution: f64,
    pub context_modifier: f64,
    pub total: f64,
}

/// Human-readable explanation of influence affecting a claim.
///
/// Includes `context_notes` when context modifiers are active.
#[derive(Debug, Clone)]
pub struct InfluenceExplanation {
    pub claim_id: ArgumentId,
    pub superiority_reasons: Vec<String>,
    pub defeater_reasons: Vec<String>,
    pub context_notes: Vec<String>,
    pub final_score: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ArgumentGraph {
    pub claims: HashMap<ArgumentId, Claim>,
    pub supports: Vec<Support>,
    pub attacks: Vec<Attack>,
    pub superiorities: Vec<Superiority>,
    pub defeaters: Vec<Defeater>,
    pub phase4_config: Phase4Config,
    next_id: ArgumentId,
}

impl ArgumentGraph {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            supports: Vec::new(),
            attacks: Vec::new(),
            superiorities: Vec::new(),
            defeaters: Vec::new(),
            phase4_config: Phase4Config::default(),
            next_id: 1,
        }
    }

    pub fn set_phase4_config(&mut self, config: Phase4Config) {
        self.phase4_config = config;
    }

    pub fn get_phase4_config(&self) -> &Phase4Config {
        &self.phase4_config
    }

    pub fn add_claim(&mut self, content: String, proposed_by: String, strength: f64) -> ArgumentId {
        let id = self.next_id;
        self.next_id += 1;
        let claim = Claim {
            id,
            content,
            proposed_by,
            strength: strength.clamp(0.0, 1.0),
            is_strict: false,
        };
        self.claims.insert(id, claim);
        id
    }

    pub fn set_strict(&mut self, claim_id: ArgumentId, is_strict: bool) {
        if let Some(claim) = self.claims.get_mut(&claim_id) {
            claim.is_strict = is_strict;
        }
    }

    pub fn add_support(
        &mut self,
        source_claim_id: ArgumentId,
        target_claim_id: ArgumentId,
        content: String,
        provided_by: String,
        strength: f64,
    ) -> Option<ArgumentId> {
        if !self.claims.contains_key(&source_claim_id) || !self.claims.contains_key(&target_claim_id) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.supports.push(Support {
            id,
            source_claim_id,
            target_claim_id,
            content,
            strength: strength.clamp(0.0, 1.0),
            provided_by,
        });
        Some(id)
    }

    pub fn add_attack(
        &mut self,
        source_claim_id: ArgumentId,
        target_claim_id: ArgumentId,
        content: String,
        provided_by: String,
        strength: f64,
    ) -> Option<ArgumentId> {
        if !self.claims.contains_key(&source_claim_id) || !self.claims.contains_key(&target_claim_id) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.attacks.push(Attack {
            id,
            source_claim_id,
            target_claim_id,
            content,
            strength: strength.clamp(0.0, 1.0),
            provided_by,
        });
        Some(id)
    }

    pub fn add_defeater(
        &mut self,
        source_claim_id: ArgumentId,
        target_claim_id: ArgumentId,
        strength: Option<f64>,
        provided_by: String,
        reason: String,
        context: Option<SuperiorityContext>,
    ) -> Option<ArgumentId> {
        if !self.claims.contains_key(&source_claim_id) || !self.claims.contains_key(&target_claim_id) {
            return None;
        }

        let final_strength = strength.unwrap_or_else(|| {
            self.claims.get(&source_claim_id).map(|c| c.strength).unwrap_or(0.5)
        });

        let id = self.next_id;
        self.next_id += 1;

        self.defeaters.push(Defeater {
            id,
            source_claim_id,
            target_claim_id,
            strength: final_strength.clamp(0.0, 1.0),
            provided_by,
            reason,
            context,
        });

        Some(id)
    }

    pub fn get_defeaters(&self, claim_id: ArgumentId) -> Vec<&Defeater> {
        self.defeaters.iter().filter(|d| d.target_claim_id == claim_id).collect()
    }

    pub fn add_superiority(
        &mut self,
        stronger: ArgumentId,
        weaker: ArgumentId,
        context: Option<SuperiorityContext>,
    ) {
        if self.superiorities.iter().any(|s| {
            s.stronger == stronger && s.weaker == weaker && s.context == context
        }) {
            return;
        }

        if let Some(pos) = self.superiorities.iter().position(|s| {
            s.stronger == weaker && s.weaker == stronger
        }) {
            self.superiorities.remove(pos);
            eprintln!(
                "[Warning] Superiority conflict resolved by recency: {} > {} (previous reverse removed)",
                stronger, weaker
            );
        }

        self.superiorities.push(Superiority {
            stronger,
            weaker,
            context,
        });
    }

    pub fn is_superior(
        &self,
        a: ArgumentId,
        b: ArgumentId,
        context: Option<SuperiorityContext>,
    ) -> bool {
        self.superiorities.iter().any(|s| {
            s.stronger == a && s.weaker == b && s.context == context
        })
    }

    pub fn get_superiorities(&self) -> Vec<Superiority> {
        self.superiorities.clone()
    }

    pub fn load_superiorities(&mut self, superiorities: Vec<Superiority>) {
        self.superiorities = superiorities;
    }

    pub fn get_supporters(&self, claim_id: ArgumentId) -> Vec<&Support> {
        self.supports.iter().filter(|s| s.target_claim_id == claim_id).collect()
    }

    pub fn get_attackers(&self, claim_id: ArgumentId) -> Vec<&Attack> {
        self.attacks.iter().filter(|a| a.target_claim_id == claim_id).collect()
    }

    pub fn get_supported_claims(&self, claim_id: ArgumentId) -> Vec<ArgumentId> {
        self.supports.iter().filter(|s| s.source_claim_id == claim_id).map(|s| s.target_claim_id).collect()
    }

    pub fn get_attacked_claims(&self, claim_id: ArgumentId) -> Vec<ArgumentId> {
        self.attacks.iter().filter(|a| a.source_claim_id == claim_id).map(|a| a.target_claim_id).collect()
    }

    pub fn conflict_level(&self, claim_id: ArgumentId) -> Option<f64> {
        let support_count = self.supports.iter().filter(|s| s.target_claim_id == claim_id).count() as f64;
        let attack_count = self.attacks.iter().filter(|a| a.target_claim_id == claim_id).count() as f64;
        if support_count + attack_count == 0.0 { return Some(0.0); }
        Some(attack_count / (support_count + attack_count))
    }

    pub fn overall_conflict_score(&self) -> f64 {
        if self.claims.is_empty() { return 0.0; }
        let total: f64 = self.claims.keys().filter_map(|&id| self.conflict_level(id)).sum();
        total / self.claims.len() as f64
    }

    pub fn most_contested_claim(&self) -> Option<(ArgumentId, f64)> {
        self.claims.keys()
            .filter_map(|&id| self.conflict_level(id).map(|level| (id, level)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn graph_summary(&self) -> (usize, usize, usize) {
        (self.claims.len(), self.supports.len(), self.attacks.len())
    }

    pub fn effective_strength(&self, claim_id: ArgumentId) -> Option<f64> {
        let base = self.claims.get(&claim_id)?.strength;
        let mut score = base;

        for support in &self.supports {
            if support.target_claim_id == claim_id { score += support.strength * 0.35; }
        }
        for attack in &self.attacks {
            if attack.target_claim_id == claim_id { score -= attack.strength * 0.45; }
        }

        let defeaters: Vec<&Defeater> = self.defeaters.iter()
            .filter(|d| d.target_claim_id == claim_id)
            .collect();

        if !defeaters.is_empty() {
            let mut defeat_impact = 0.0;
            for (i, d) in defeaters.iter().enumerate() {
                let factor = if i == 0 { 1.0 } else { 0.6f64.powf(i as f64) };
                defeat_impact += d.strength * factor;
            }
            score -= defeat_impact * 0.5;
        }

        Some(score.clamp(0.0, 1.0))
    }

    pub fn ranked_claims(&self) -> Vec<(ArgumentId, f64)> {
        let mut ranked: Vec<(ArgumentId, f64)> = self.claims.keys()
            .filter_map(|&id| self.effective_strength(id).map(|s| (id, s))).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    pub fn calculate_influence_score(&self, claim_id: ArgumentId) -> InfluenceScore {
        let mut score = InfluenceScore::default();

        if !self.phase4_config.enable_extension_influence {
            return score;
        }

        for sup in &self.superiorities {
            if sup.stronger == claim_id {
                let weight = self.context_weight(&sup.context);
                score.superiority_contribution += 0.15 * weight;
            }
            if sup.weaker == claim_id {
                let weight = self.context_weight(&sup.context);
                score.superiority_contribution -= 0.12 * weight;
            }
        }

        for def in &self.defeaters {
            if def.target_claim_id == claim_id {
                let mut impact = def.strength * 0.5;

                if self.phase4_config.enable_defeater_context_modifiers {
                    let ctx = def.context.clone();
                    impact = self.apply_defeater_context_modifier(impact, &ctx);
                }

                score.defeater_contribution -= impact;
            }
        }

        let raw_total = score.superiority_contribution + score.defeater_contribution + score.context_modifier;
        score.total = raw_total.clamp(-self.phase4_config.max_influence_strength, self.phase4_config.max_influence_strength);

        if self.phase4_config.enable_influence_logging {
            eprintln!(
                "[Phase4] Influence on claim {}: sup={:.3}, def={:.3}, ctx_mod={:.3}, total={:.3}",
                claim_id,
                score.superiority_contribution,
                score.defeater_contribution,
                score.context_modifier,
                score.total
            );
        }

        score
    }

    /// Applies context-specific modification to a defeater's negative impact.
    ///
    /// Special behaviors:
    /// - `MercyGate`: Reduces impact (compassionate softening)
    /// - `Council`: Increases impact (authoritative / decisive weight)
    fn apply_defeater_context_modifier(&self, base_impact: f64, context: &Option<SuperiorityContext>) -> f64 {
        match context {
            Some(SuperiorityContext::MercyGate) => {
                // Mercy-aligned reduction
                base_impact * 0.6
            }
            Some(SuperiorityContext::Council) => {
                // Authoritative / structural weight
                base_impact * 1.25
            }
            Some(ctx) => {
                let weight = self.context_weight(&Some(ctx.clone()));
                base_impact * weight
            }
            None => base_impact,
        }
    }

    pub fn explain_influence(&self, claim_id: ArgumentId) -> InfluenceExplanation {
        let mut explanation = InfluenceExplanation {
            claim_id,
            superiority_reasons: vec![],
            defeater_reasons: vec![],
            context_notes: vec![],
            final_score: 0.0,
        };

        if !self.phase4_config.enable_extension_influence {
            explanation.final_score = 0.0;
            return explanation;
        }

        let score = self.calculate_influence_score(claim_id);
        explanation.final_score = score.total;

        for sup in &self.superiorities {
            if sup.stronger == claim_id {
                explanation.superiority_reasons.push(format!("Supported by superiority from {}", sup.weaker));
            }
            if sup.weaker == claim_id {
                explanation.superiority_reasons.push(format!("Weakened by superiority from {}", sup.stronger));
            }
        }

        for def in &self.defeaters {
            if def.target_claim_id == claim_id {
                let mut reason = if def.reason.is_empty() {
                    format!("Weakened by defeater from {} (strength {:.2})", def.source_claim_id, def.strength)
                } else {
                    def.reason.clone()
                };

                if self.phase4_config.enable_defeater_context_modifiers {
                    match &def.context {
                        Some(SuperiorityContext::MercyGate) => {
                            explanation.context_notes.push("MercyGate reduction applied (defeater softened)".to_string());
                        }
                        Some(SuperiorityContext::Council) => {
                            explanation.context_notes.push("Council context applied (authoritative weight)".to_string());
                        }
                        Some(ctx) => {
                            explanation.context_notes.push(format!("Context {:?} applied", ctx));
                        }
                        None => {}
                    }
                }
                explanation.defeater_reasons.push(reason);
            }
        }

        explanation
    }

    pub fn preferred_extensions_with_influence(&self, max_results: usize) -> Vec<HashSet<ArgumentId>> {
        if !self.phase4_config.enable_extension_influence {
            return self.preferred_extensions(max_results);
        }

        let pure_extensions = self.preferred_extensions(max_results * 2);
        let mut influenced: Vec<(HashSet<ArgumentId>, f64)> = Vec::new();

        for ext in pure_extensions {
            let mut total_influence: f64 = 0.0;

            for &arg in &ext {
                let influence = self.calculate_influence_score(arg);
                total_influence += influence.total;
            }

            influenced.push((ext, total_influence));
        }

        influenced.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.phase4_config.enable_influence_logging {
            eprintln!("[Phase4] Re-ranked {} extensions using influence scores.", influenced.len());
        }

        influenced.into_iter().map(|(ext, _)| ext).take(max_results).collect()
    }

    pub fn unattacked_arguments(&self) -> Vec<ArgumentId> {
        self.claims.keys().filter(|&&id| self.get_attackers(id).is_empty()).cloned().collect()
    }

    fn is_defended(&self, claim_id: ArgumentId, defeated: &HashSet<ArgumentId>) -> bool {
        self.get_attackers(claim_id).iter().all(|attack| defeated.contains(&attack.source_claim_id))
    }

    pub fn is_admissible(&self, set: &HashSet<ArgumentId>) -> bool {
        for &arg in set {
            for attack in self.get_attackers(arg) {
                if set.contains(&attack.source_claim_id) { return false; }
            }
        }
        for &arg in set {
            if !self.is_defended(arg, set) { return false; }
        }
        true
    }

    pub fn grounded_extension(&self) -> Vec<ArgumentId> {
        let mut extension: HashSet<ArgumentId> = HashSet::new();
        let mut changed = true;
        let mut to_check: HashSet<ArgumentId> = self.unattacked_arguments().into_iter().collect();

        while changed {
            changed = false;
            let mut newly_accepted = Vec::new();

            for &arg in &to_check {
                if self.is_defended(arg, &extension) && !extension.contains(&arg) {
                    newly_accepted.push(arg);
                }
            }

            for arg in newly_accepted {
                extension.insert(arg);
                changed = true;
            }

            to_check = self.claims.keys()
                .filter(|&&id| !extension.contains(&id))
                .filter(|&&id| self.is_defended(id, &extension))
                .cloned()
                .collect();
        }
        extension.into_iter().collect()
    }

    pub fn find_maximal_admissible_set(&self) -> HashSet<ArgumentId> {
        let mut current: HashSet<ArgumentId> = self.grounded_extension().into_iter().collect();
        for &arg in self.claims.keys() {
            if current.contains(&arg) { continue; }
            let mut test_set = current.clone();
            test_set.insert(arg);
            if self.is_admissible(&test_set) {
                current = test_set;
            }
        }
        current
    }

    pub fn preferred_extensions(&self, max_results: usize) -> Vec<HashSet<ArgumentId>> {
        let mut results: Vec<HashSet<ArgumentId>> = Vec::new();
        let grounded: HashSet<ArgumentId> = self.grounded_extension().into_iter().collect();
        let mut stack: Vec<HashSet<ArgumentId>> = vec![grounded];

        while let Some(current) = stack.pop() {
            if results.len() >= max_results { break; }
            let mut extended = false;
            for &arg in self.claims.keys() {
                if current.contains(&arg) { continue; }
                let mut test_set = current.clone();
                test_set.insert(arg);
                if self.is_admissible(&test_set) {
                    stack.push(test_set);
                    extended = true;
                    break;
                }
            }
            if !extended {
                if !results.iter().any(|r| r == &current) {
                    results.push(current);
                }
            }
        }
        results
    }

    pub fn is_stable(&self, set: &HashSet<ArgumentId>) -> bool {
        if !self.is_admissible(set) { return false; }
        for &arg in self.claims.keys() {
            if set.contains(&arg) { continue; }
            let mut attacks_outside = false;
            for attack in self.get_attackers(arg) {
                if set.contains(&attack.source_claim_id) {
                    attacks_outside = true;
                    break;
                }
            }
            if !attacks_outside { return false; }
        }
        true
    }

    pub fn stable_extensions(&self, max_results: usize) -> Vec<HashSet<ArgumentId>> {
        let mut results = Vec::new();
        let preferreds = self.preferred_extensions(max_results * 2);
        for pref in preferreds {
            if self.is_stable(&pref) {
                if !results.iter().any(|r| r == &pref) {
                    results.push(pref);
                }
                if results.len() >= max_results { break; }
            }
        }
        results
    }

    pub fn recommend_extensions(&self) -> ExtensionRecommendation {
        let grounded = self.grounded_extension();
        let preferreds = self.preferred_extensions(3);
        let stables = self.stable_extensions(2);

        let strict_in_grounded = grounded.iter()
            .filter_map(|&id| self.claims.get(&id))
            .filter(|c| c.is_strict)
            .count();

        let total_grounded = grounded.len().max(1) as f64;
        let strict_ratio = strict_in_grounded as f64 / total_grounded;

        let mut context_weighted_bonus = 0.0;
        for sup in &self.superiorities {
            let weight = self.context_weight(&sup.context);
            context_weighted_bonus += 0.08 * weight;
        }

        let defeater_penalty = if self.defeaters.is_empty() { 0.0 } else {
            let mut penalty = 0.0;
            for d in &self.defeaters {
                let weight = self.context_weight(&d.context);
                penalty += 0.06 * weight;
            }
            (penalty / self.claims.len().max(1) as f64).min(0.25)
        };

        let base_safety = if self.claims.is_empty() { 0.0 } else {
            grounded.len() as f64 / self.claims.len() as f64
        };

        let base_evolution = if preferreds.is_empty() && stables.is_empty() {
            0.1
        } else {
            let pref_avg = if preferreds.is_empty() { 0.0 } else {
                preferreds.iter().map(|p| p.len()).sum::<usize>() as f64 / (preferreds.len() as f64 * self.claims.len() as f64)
            };
            let stable_avg = if stables.is_empty() { 0.0 } else {
                stables.iter().map(|s| s.len()).sum::<usize>() as f64 / (stables.len() as f64 * self.claims.len() as f64)
            };
            (pref_avg * 0.6 + stable_avg * 0.4).min(1.0)
        };

        let safety_score = ((base_safety + strict_ratio * 0.25) * (1.0 - defeater_penalty)).clamp(0.0, 1.0);
        let evolution_potential = (base_evolution + context_weighted_bonus.min(0.25)).min(1.0);

        let overall_score = (safety_score * 0.55) + (evolution_potential * 0.45);

        let recommendation = if evolution_potential > 0.55 {
            "Good evolution potential with context-aware considerations."
        } else if safety_score > 0.65 {
            "Strong safety posture with strict claims."
        } else {
            "Balanced position between safety and evolution."
        };

        ExtensionRecommendation {
            grounded,
            preferreds,
            stables,
            safety_score,
            evolution_potential,
            overall_score,
            recommendation,
        }
    }

    fn context_weight(&self, context: &Option<SuperiorityContext>) -> f64 {
        match context {
            Some(SuperiorityContext::Council) => 1.25,
            Some(SuperiorityContext::Topic)   => 1.10,
            Some(SuperiorityContext::General) => 1.00,
            Some(SuperiorityContext::MercyGate) => 0.6,
            Some(SuperiorityContext::SelfEvolution) => 1.10,
            Some(SuperiorityContext::Custom(_)) => 1.05,
            None => 1.00,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtensionRecommendation {
    pub grounded: Vec<ArgumentId>,
    pub preferreds: Vec<HashSet<ArgumentId>>,
    pub stables: Vec<HashSet<ArgumentId>>,
    pub safety_score: f64,
    pub evolution_potential: f64,
    pub overall_score: f64,
    pub recommendation: String,
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

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new().with_extension_influence(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);
        assert_eq!(score.context_modifier, 0.0);
    }

    #[test]
    fn test_council_context_increases_impact_when_enabled() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);

        // Council context should increase the negative impact (authoritative weight)
        assert!(score.defeater_contribution < -0.3);
    }

    #[test]
    fn test_mercygate_reduces_impact_when_enabled() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::MercyGate));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);

        // MercyGate should reduce the negative impact
        assert!(score.defeater_contribution > -0.4);
    }

    #[test]
    fn test_self_evolution_increases_impact_when_enabled() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::SelfEvolution));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);

        // SelfEvolution should increase the negative impact
        assert!(score.defeater_contribution < -0.3);
    }

    #[test]
    fn test_explain_influence_includes_council_context() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let explanation = graph.explain_influence(target);
        let has_council_note = explanation.context_notes.iter().any(|n| n.contains("Council"));
        assert!(has_council_note);
    }

    #[test]
    fn test_explain_influence_includes_mercygate_context() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::MercyGate));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let explanation = graph.explain_influence(target);
        let has_mercygate_note = explanation.context_notes.iter().any(|n| n.contains("MercyGate"));
        assert!(has_mercygate_note);
    }

    #[test]
    fn test_explain_influence_includes_self_evolution_context() {
        let mut graph = ArgumentGraph::new();
        let source = graph.add_claim("Source".to_string(), "Test".to_string(), 0.8);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.7);

        graph.add_defeater(source, target, Some(0.6), "Test".to_string(), "".to_string(), Some(SuperiorityContext::SelfEvolution));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let explanation = graph.explain_influence(target);
        let has_self_evolution_note = explanation.context_notes.iter().any(|n| n.contains("SelfEvolution"));
        assert!(has_self_evolution_note);
    }

    #[test]
    fn test_mixed_contexts_on_same_claim() {
        let mut graph = ArgumentGraph::new();
        let source1 = graph.add_claim("Source1".to_string(), "Test".to_string(), 0.8);
        let source2 = graph.add_claim("Source2".to_string(), "Test".to_string(), 0.7);
        let target = graph.add_claim("Target".to_string(), "Test".to_string(), 0.6);

        // One MercyGate defeater (softens) and one Council defeater (strengthens)
        graph.add_defeater(source1, target, Some(0.5), "Test".to_string(), "".to_string(), Some(SuperiorityContext::MercyGate));
        graph.add_defeater(source2, target, Some(0.5), "Test".to_string(), "".to_string(), Some(SuperiorityContext::Council));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(target);

        // Net effect should still be negative, but moderated by the MercyGate reduction
        assert!(score.defeater_contribution < 0.0);
        assert!(score.context_modifier != 0.0);
    }

    #[test]
    fn test_superiority_and_mercygate_defeater_combined() {
        let mut graph = ArgumentGraph::new();
        let strong = graph.add_claim("Strong".to_string(), "Test".to_string(), 0.8);
        let weak = graph.add_claim("Weak".to_string(), "Test".to_string(), 0.6);
        let defeater_source = graph.add_claim("DefeaterSource".to_string(), "Test".to_string(), 0.7);

        // Superiority: strong > weak
        graph.add_superiority(strong, weak, Some(SuperiorityContext::General));

        // MercyGate defeater targeting weak
        graph.add_defeater(defeater_source, weak, Some(0.5), "Test".to_string(), "".to_string(), Some(SuperiorityContext::MercyGate));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score = graph.calculate_influence_score(weak);

        // Should have both superiority contribution and softened defeater contribution
        assert!(score.superiority_contribution > 0.0);
        assert!(score.defeater_contribution < 0.0);
    }

    #[test]
    fn test_multiple_superiorities_with_different_contexts() {
        let mut graph = ArgumentGraph::new();
        let claim_a = graph.add_claim("ClaimA".to_string(), "Test".to_string(), 0.7);
        let claim_b = graph.add_claim("ClaimB".to_string(), "Test".to_string(), 0.6);
        let claim_c = graph.add_claim("ClaimC".to_string(), "Test".to_string(), 0.5);

        // claim_a is superior in a Council context
        graph.add_superiority(claim_a, claim_b, Some(SuperiorityContext::Council));

        // claim_b is superior in a general context
        graph.add_superiority(claim_b, claim_c, Some(SuperiorityContext::General));

        let config = Phase4Config::new()
            .with_extension_influence(true)
            .with_defeater_context_modifiers(true);
        graph.set_phase4_config(config);

        let score_b = graph.calculate_influence_score(claim_b);
        let score_c = graph.calculate_influence_score(claim_c);

        // claim_b should receive positive influence from Council superiority
        assert!(score_b.superiority_contribution > 0.0);

        // claim_c should receive positive influence (weaker context)
        assert!(score_c.superiority_contribution > 0.0);
    }
}