// crates/lattice-conductor-v14/src/logical_fallacy_detection.rs
// Updated to fully leverage improved ArgumentGraph claim-to-claim model

use crate::argumentation::{ArgumentGraph, ArgumentId};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FallacyType {
    UnsupportedClaim,
    CircularSupport,
    WeakEvidenceChain,
}

#[derive(Debug, Clone)]
pub struct DetectedFallacy {
    pub fallacy_type: FallacyType,
    pub target_claim_id: ArgumentId,
    pub description: String,
    pub severity: f64,
}

pub struct LogicalFallacyDetector;

impl LogicalFallacyDetector {
    pub fn detect_structural_fallacies(graph: &ArgumentGraph) -> Vec<DetectedFallacy> {
        let mut fallacies = Vec::new();

        // Unsupported claims
        for (id, claim) in &graph.claims {
            let has_support = graph.supports.iter().any(|s| s.target_claim_id == *id);
            if !has_support {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::UnsupportedClaim,
                    target_claim_id: *id,
                    description: format!("Unsupported claim: {}", claim.content),
                    severity: 0.6,
                });
            }
        }

        // Improved circular support detection
        let circular = Self::detect_circular_support(graph);
        for claim_id in circular {
            if let Some(claim) = graph.claims.get(&claim_id) {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::CircularSupport,
                    target_claim_id: claim_id,
                    description: format!("Circular support detected: {}", claim.content),
                    severity: 0.92,
                });
            }
        }

        // Weak evidence chain
        for (id, claim) in &graph.claims {
            let support_count = graph.supports.iter().filter(|s| s.target_claim_id == *id).count();
            let attack_count = graph.attacks.iter().filter(|a| a.target_claim_id == *id).count();

            if attack_count > support_count + 1 {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::WeakEvidenceChain,
                    target_claim_id: *id,
                    description: format!("Weak evidence chain: {}", claim.content),
                    severity: 0.7,
                });
            }
        }

        fallacies
    }

    /// Proper circular detection using claim-to-claim support graph
    pub fn detect_circular_support(graph: &ArgumentGraph) -> HashSet<ArgumentId> {
        // Build adjacency list: claim -> list of claims it supports
        let mut adj: HashMap<ArgumentId, Vec<ArgumentId>> = HashMap::new();

        for support in &graph.supports {
            adj.entry(support.source_claim_id)
                .or_default()
                .push(support.target_claim_id);
        }

        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut circular = HashSet::new();

        for &claim_id in graph.claims.keys() {
            if !visited.contains(&claim_id) {
                if Self::dfs_has_cycle(claim_id, &adj, &mut visited, &mut rec_stack) {
                    circular.insert(claim_id);
                }
            }
        }

        circular
    }

    fn dfs_has_cycle(
        node: ArgumentId,
        adj: &HashMap<ArgumentId, Vec<ArgumentId>>,
        visited: &mut HashSet<ArgumentId>,
        rec_stack: &mut HashSet<ArgumentId>,
    ) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        if let Some(neighbors) = adj.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if Self::dfs_has_cycle(neighbor, adj, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&neighbor) {
                    return true;
                }
            }
        }

        rec_stack.remove(&node);
        false
    }
}
