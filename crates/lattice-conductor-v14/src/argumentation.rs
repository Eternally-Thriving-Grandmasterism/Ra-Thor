// crates/lattice-conductor-v14/src/argumentation.rs
//
// Ra-Thor Argumentation Graph
//
// This module provides a flexible argument graph supporting both practical analysis
// and formal Abstract Argumentation Framework semantics (inspired by Dung's theory).
//
// Key Capabilities:
// - Claim, Support, and Attack modeling
// - Conflict and strength analysis
// - Formal semantics: Grounded, Preferred, and Stable Extensions
// - Numeric recommendation scoring (Safety vs Evolution Potential)
//
// This system is designed to support PATSAGi Council decision-making with
// a balance of truth, mercy, and evolution.

use std::collections::{HashMap, HashSet};

pub type ArgumentId = u64;

#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String,
    pub strength: f64,
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

#[derive(Debug, Clone, Default)]
pub struct ArgumentGraph {
    pub claims: HashMap<ArgumentId, Claim>,
    pub supports: Vec<Support>,
    pub attacks: Vec<Attack>,
    next_id: ArgumentId,
}

impl ArgumentGraph {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            supports: Vec::new(),
            attacks: Vec::new(),
            next_id: 1,
        }
    }

    pub fn add_claim(&mut self, content: String, proposed_by: String, strength: f64) -> ArgumentId {
        let id = self.next_id;
        self.next_id += 1;
        let claim = Claim { id, content, proposed_by, strength: strength.clamp(0.0, 1.0) };
        self.claims.insert(id, claim);
        id
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
        Some(score.clamp(0.0, 1.0))
    }

    pub fn ranked_claims(&self) -> Vec<(ArgumentId, f64)> {
        let mut ranked: Vec<(ArgumentId, f64)> = self.claims.keys()
            .filter_map(|&id| self.effective_strength(id).map(|s| (id, s))).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // === Formal Abstract Argumentation Semantics ===

    /// Returns arguments with no attackers.
    pub fn unattacked_arguments(&self) -> Vec<ArgumentId> {
        self.claims
            .keys()
            .filter(|&&id| self.get_attackers(id).is_empty())
            .cloned()
            .collect()
    }

    fn is_defended(&self, claim_id: ArgumentId, defeated: &HashSet<ArgumentId>) -> bool {
        self.get_attackers(claim_id)
            .iter()
            .all(|attack| defeated.contains(&attack.source_claim_id))
    }

    /// Checks if a set is admissible (conflict-free and defends itself).
    pub fn is_admissible(&self, set: &HashSet<ArgumentId>) -> bool {
        for &arg in set {
            for attack in self.get_attackers(arg) {
                if set.contains(&attack.source_claim_id) {
                    return false;
                }
            }
        }
        for &arg in set {
            if !self.is_defended(arg, set) {
                return false;
            }
        }
        true
    }

    /// Computes the Grounded Extension (most skeptical, well-founded position).
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

    /// Returns up to `max_results` Preferred Extensions (maximal admissible sets).
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

    /// Checks if a set is a Stable Extension.
    pub fn is_stable(&self, set: &HashSet<ArgumentId>) -> bool {
        if !self.is_admissible(set) {
            return false;
        }

        for &arg in self.claims.keys() {
            if set.contains(&arg) { continue; }

            let mut attacks_outside = false;
            for attack in self.get_attackers(arg) {
                if set.contains(&attack.source_claim_id) {
                    attacks_outside = true;
                    break;
                }
            }
            if !attacks_outside {
                return false;
            }
        }
        true
    }

    /// Returns up to `max_results` Stable Extensions.
    pub fn stable_extensions(&self, max_results: usize) -> Vec<HashSet<ArgumentId>> {
        let mut results = Vec::new();
        let preferreds = self.preferred_extensions(max_results * 2);

        for pref in preferreds {
            if self.is_stable(&pref) {
                if !results.iter().any(|r| r == &pref) {
                    results.push(pref);
                }
                if results.len() >= max_results {
                    break;
                }
            }
        }
        results
    }

    // === Recommendation Engine ===

    /// Returns a structured recommendation with numeric Safety and Evolution scores.
    ///
    /// # Example
    /// ```
    /// use lattice_conductor_v14::ArgumentGraph;
    ///
    /// let mut graph = ArgumentGraph::new();
    /// let claim_id = graph.add_claim("Self-evolution is needed".to_string(), "Council #13".to_string(), 0.85);
    ///
    /// let rec = graph.recommend_extensions();
    /// println!("Safety Score: {}", rec.safety_score);
    /// println!("Evolution Potential: {}", rec.evolution_potential);
    /// println!("Recommendation: {}", rec.recommendation);
    /// ```
    pub fn recommend_extensions(&self) -> ExtensionRecommendation {
        let grounded = self.grounded_extension();
        let preferreds = self.preferred_extensions(3);
        let stables = self.stable_extensions(2);

        let safety_score = if self.claims.is_empty() { 0.0 } else {
            grounded.len() as f64 / self.claims.len() as f64
        };

        let evolution_potential = if preferreds.is_empty() && stables.is_empty() {
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

        let overall_score = (safety_score * 0.55) + (evolution_potential * 0.45);

        let recommendation = if evolution_potential > 0.5 {
            "Good evolution potential exists while maintaining reasonable safety."
        } else if safety_score > 0.6 {
            "Strong safety-focused position available (Grounded)."
        } else {
            "Balanced consideration between safety and evolution is recommended."
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
