// crates/lattice-conductor-v14/src/argumentation.rs
// Phase 1 Polish: Clean, well-documented ArgumentGraph
// Prepared for future formal Abstract Argumentation semantics

use std::collections::HashMap;

pub type ArgumentId = u64;

/// A single claim or position in an argument graph
#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String,
    pub strength: f64,
}

/// Represents support from one claim to another
#[derive(Debug, Clone)]
pub struct Support {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// Represents an attack from one claim against another
#[derive(Debug, Clone)]
pub struct Attack {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// A directed graph of claims, supports, and attacks.
/// Designed to support future formal argumentation semantics (e.g. Dung-style).
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

    /// Add a new claim to the graph
    pub fn add_claim(&mut self, content: String, proposed_by: String, strength: f64) -> ArgumentId {
        let id = self.next_id;
        self.next_id += 1;
        let claim = Claim { id, content, proposed_by, strength: strength.clamp(0.0, 1.0) };
        self.claims.insert(id, claim);
        id
    }

    /// Add a support relation from one claim to another
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

    /// Add an attack relation from one claim against another
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

    /// Get all supports targeting a specific claim
    pub fn get_supporters(&self, claim_id: ArgumentId) -> Vec<&Support> {
        self.supports.iter().filter(|s| s.target_claim_id == claim_id).collect()
    }

    /// Get all attacks targeting a specific claim
    pub fn get_attackers(&self, claim_id: ArgumentId) -> Vec<&Attack> {
        self.attacks.iter().filter(|a| a.target_claim_id == claim_id).collect()
    }

    /// Get claims supported by a given claim
    pub fn get_supported_claims(&self, claim_id: ArgumentId) -> Vec<ArgumentId> {
        self.supports.iter()
            .filter(|s| s.source_claim_id == claim_id)
            .map(|s| s.target_claim_id)
            .collect()
    }

    /// Get claims attacked by a given claim
    pub fn get_attacked_claims(&self, claim_id: ArgumentId) -> Vec<ArgumentId> {
        self.attacks.iter()
            .filter(|a| a.source_claim_id == claim_id)
            .map(|a| a.target_claim_id)
            .collect()
    }

    /// Conflict level of a claim (ratio of attacks to total relations)
    pub fn conflict_level(&self, claim_id: ArgumentId) -> Option<f64> {
        let support_count = self.supports.iter().filter(|s| s.target_claim_id == claim_id).count() as f64;
        let attack_count = self.attacks.iter().filter(|a| a.target_claim_id == claim_id).count() as f64;
        if support_count + attack_count == 0.0 { return Some(0.0); }
        Some(attack_count / (support_count + attack_count))
    }

    /// Average conflict level across all claims in the graph
    pub fn overall_conflict_score(&self) -> f64 {
        if self.claims.is_empty() { return 0.0; }
        let total: f64 = self.claims.keys().filter_map(|&id| self.conflict_level(id)).sum();
        total / self.claims.len() as f64
    }

    /// Returns the claim with the highest conflict level, if any
    pub fn most_contested_claim(&self) -> Option<(ArgumentId, f64)> {
        self.claims.keys()
            .filter_map(|&id| self.conflict_level(id).map(|level| (id, level)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Quick summary: (total_claims, total_supports, total_attacks)
    pub fn graph_summary(&self) -> (usize, usize, usize) {
        (self.claims.len(), self.supports.len(), self.attacks.len())
    }

    /// Calculate effective strength of a claim after supports and attacks
    pub fn effective_strength(&self, claim_id: ArgumentId) -> Option<f64> {
        let base = self.claims.get(&claim_id)?.strength;
        let mut score = base;

        for support in &self.supports {
            if support.target_claim_id == claim_id {
                score += support.strength * 0.35;
            }
        }
        for attack in &self.attacks {
            if attack.target_claim_id == claim_id {
                score -= attack.strength * 0.45;
            }
        }
        Some(score.clamp(0.0, 1.0))
    }

    /// Returns claims ranked by effective strength (highest first)
    pub fn ranked_claims(&self) -> Vec<(ArgumentId, f64)> {
        let mut ranked: Vec<(ArgumentId, f64)> = self
            .claims
            .keys()
            .filter_map(|&id| self.effective_strength(id).map(|s| (id, s)))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}
