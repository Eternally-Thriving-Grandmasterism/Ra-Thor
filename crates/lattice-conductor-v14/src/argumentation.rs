// crates/lattice-conductor-v14/src/argumentation.rs
//
// Ra-Thor Argumentation Graph
// Phase 1: Defeasible Logic Integration (Contextual Superiority + Strict/Defeasible Claims)
//
// This phase introduces foundational defeasible logic concepts:
// - Claims can be marked as strict (cannot be defeated) or defeasible
// - Superiority relations between arguments (contextual)
// - These concepts currently influence the Recommendation Engine

use std::collections::{HashMap, HashSet};

pub type ArgumentId = u64;

#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String,
    pub strength: f64,
    pub is_strict: bool, // NEW in Phase 1: true = cannot be defeated
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

/// Represents a superiority relation between two arguments.
/// `context` allows superiority to be scoped (e.g. "Council13", "Safety", topic-based).
#[derive(Debug, Clone)]
pub struct Superiority {
    pub stronger: ArgumentId,
    pub weaker: ArgumentId,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ArgumentGraph {
    pub claims: HashMap<ArgumentId, Claim>,
    pub supports: Vec<Support>,
    pub attacks: Vec<Attack>,
    pub superiorities: Vec<Superiority>, // NEW in Phase 1
    next_id: ArgumentId,
}

impl ArgumentGraph {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            supports: Vec::new(),
            attacks: Vec::new(),
            superiorities: Vec::new(),
            next_id: 1,
        }
    }

    pub fn add_claim(&mut self, content: String, proposed_by: String, strength: f64) -> ArgumentId {
        let id = self.next_id;
        self.next_id += 1;
        let claim = Claim {
            id,
            content,
            proposed_by,
            strength: strength.clamp(0.0, 1.0),
            is_strict: false, // Default: defeasible
        };
        self.claims.insert(id, claim);
        id
    }

    /// Set whether a claim is strict (cannot be defeated) or defeasible.
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

    /// Declare that one argument is superior to another.
    /// `context` is optional and allows scoped superiority (e.g. topic or council-specific).
    /// If a conflicting superiority already exists, we keep the first declaration (Phase 1 behavior).
    pub fn add_superiority(
        &mut self,
        stronger: ArgumentId,
        weaker: ArgumentId,
        context: Option<&str>,
    ) {
        // Phase 1: Simple conflict handling - keep first declaration
        let already_exists = self.superiorities.iter().any(|s| {
            s.stronger == stronger && s.weaker == weaker && s.context == context.map(|c| c.to_string())
        });

        if already_exists {
            return; // Already declared
        }

        // Check for reverse superiority (potential conflict)
        let reverse_exists = self.superiorities.iter().any(|s| {
            s.stronger == weaker && s.weaker == stronger
        });

        if reverse_exists {
            eprintln!("[Warning] Conflicting superiority declared between {} and {}", stronger, weaker);
            return;
        }

        self.superiorities.push(Superiority {
            stronger,
            weaker,
            context: context.map(|c| c.to_string()),
        });
    }

    /// Check if argument `a` is superior to `b` (optionally within a context).
    pub fn is_superior(&self, a: ArgumentId, b: ArgumentId, context: Option<&str>) -> bool {
        self.superiorities.iter().any(|s| {
            s.stronger == a &&
            s.weaker == b &&
            (context.is_none() || s.context.as_deref() == context)
        })
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

    // === Formal Semantics (unchanged in Phase 1) ===

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

    pub fn grounded_extension(&self) -> Vec<ArgumentId> {
        // ... (implementation unchanged in Phase 1)
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

    // ... (other formal methods remain unchanged in Phase 1)

    pub fn recommend_extensions(&self) -> ExtensionRecommendation {
        // Placeholder - will be enhanced in next step to use superiority and is_strict
        let grounded = self.grounded_extension();
        let preferreds = self.preferred_extensions(3);
        let stables = self.stable_extensions(2);

        ExtensionRecommendation {
            grounded,
            preferreds,
            stables,
            safety_score: 0.0,
            evolution_potential: 0.0,
            overall_score: 0.0,
            recommendation: "Phase 1 superiority integration in progress".to_string(),
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
