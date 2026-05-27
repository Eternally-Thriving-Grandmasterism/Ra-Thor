// crates/lattice-conductor-v14/src/argumentation.rs
// Production-grade Argumentation Framework for PATSAGi Multi-Agent Debate
//
// Supports structured claims, support, attacks, and basic evaluation.
// Designed for integration with PATSAGi Council archetypes and governance risk systems.

use std::collections::HashMap;

/// Unique identifier for arguments
pub type ArgumentId = u64;

/// Represents a single claim or position in a debate
#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ArgumentId,
    pub content: String,
    pub proposed_by: String, // e.g., "Mercy Council", "Council #13"
    pub strength: f64,       // Base strength of the claim (0.0 - 1.0)
}

/// Represents supporting evidence or reasoning for a claim
#[derive(Debug, Clone)]
pub struct Support {
    pub id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// Represents an attack or rebuttal against a claim
#[derive(Debug, Clone)]
pub struct Attack {
    pub id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub content: String,
    pub strength: f64,
    pub provided_by: String,
}

/// A complete argument graph for a debate or decision process
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

        let claim = Claim {
            id,
            content,
            proposed_by,
            strength: strength.clamp(0.0, 1.0),
        };

        self.claims.insert(id, claim);
        id
    }

    /// Add support to an existing claim
    pub fn add_support(&mut self, target_claim_id: ArgumentId, content: String, provided_by: String, strength: f64) -> Option<ArgumentId> {
        if !self.claims.contains_key(&target_claim_id) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.supports.push(Support {
            id,
            target_claim_id,
            content,
            strength: strength.clamp(0.0, 1.0),
            provided_by,
        });

        Some(id)
    }

    /// Add an attack against an existing claim
    pub fn add_attack(&mut self, target_claim_id: ArgumentId, content: String, provided_by: String, strength: f64) -> Option<ArgumentId> {
        if !self.claims.contains_key(&target_claim_id) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.attacks.push(Attack {
            id,
            target_claim_id,
            content,
            strength: strength.clamp(0.0, 1.0),
            provided_by,
        });

        Some(id)
    }

    /// Calculate effective strength of a claim after supports and attacks
    pub fn effective_strength(&self, claim_id: ArgumentId) -> Option<f64> {
        let base_claim = self.claims.get(&claim_id)?;
        let mut effective = base_claim.strength;

        // Apply supports
        for support in &self.supports {
            if support.target_claim_id == claim_id {
                effective += support.strength * 0.3; // Support has moderate impact
            }
        }

        // Apply attacks (reduce strength)
        for attack in &self.attacks {
            if attack.target_claim_id == claim_id {
                effective -= attack.strength * 0.4; // Attacks are stronger
            }
        }

        Some(effective.clamp(0.0, 1.0))
    }

    /// Get all claims sorted by effective strength (strongest first)
    pub fn ranked_claims(&self) -> Vec<(ArgumentId, f64)> {
        let mut ranked: Vec<(ArgumentId, f64)> = self
            .claims
            .keys()
            .filter_map(|&id| self.effective_strength(id).map(|s| (id, s)))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}
