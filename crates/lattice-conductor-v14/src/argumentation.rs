// crates/lattice-conductor-v14/src/argumentation.rs
// Enhanced Production-grade ArgumentGraph with proper claim-to-claim relationships

use std::collections::HashMap;

pub type ArgumentId = u64;

#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String,
    pub strength: f64,
}

/// Support now explicitly links one claim supporting another
#[derive(Debug, Clone)]
pub struct Support {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,   // The claim providing support
    pub target_claim_id: ArgumentId,   // The claim being supported
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// Attack now explicitly links one claim attacking another
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

        let claim = Claim {
            id,
            content,
            proposed_by,
            strength: strength.clamp(0.0, 1.0),
        };
        self.claims.insert(id, claim);
        id
    }

    /// Add support from one claim to another
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

    /// Add attack from one claim against another
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

    /// Calculate effective strength considering supports and attacks
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
